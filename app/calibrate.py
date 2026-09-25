"""实播时钟偏移校准核心逻辑（纯标准库实现，不依赖 Web 框架）。

规则概述
--------
* 计划侧是与 ``POST /compile`` 相同的口令数组，复用 ``app.compiler`` 的锚点解析
  得到每条口令的计划起止毫秒；实播记录复用 ``app.reconcile.validate_records``
  的字段级校验（重复 id、空 id / 空通道、非法区间）。
* 与对账不同，校准要求实播记录与计划口令 **同 id 同通道**：记录 id 必须出现在
  计划中，且通道须与对应计划口令一致，否则返回结构化 422。
* 每个有实播记录的通道独立估计一个 **整数时钟偏移** ``offset_ms``：一条记录仅在
  「计划起点 + offset」与「计划终点 + offset」相对实际起止的偏差都不超过容差
  （含等于）时计为 **一致**。等价地，每条记录给出一个使其一致的整数偏移闭区间
  ``[max(起点偏差, 终点偏差) - 容差, min(起点偏差, 终点偏差) + 容差]``。
* 选择被最多记录区间覆盖的整数偏移；并列时先取 **绝对值最小**、再取 **数值较小**
  者。某通道所有记录在任何偏移下都无法满足双端容差（区间均为空，如实播时长与
  计划时长之差超过 2 倍容差）时，该通道明确标为 **无法校准**。
"""

from __future__ import annotations

from typing import Any

from .compiler import CompileError, compile_commands
from .reconcile import validate_records

# 通道内单条记录统一为 (偏移区间下界, 偏移区间上界, 记录 id) 元组
OffsetInterval = tuple[int, int, str]


class CalibrateError(CompileError):
    """时钟校准失败的业务异常；复用 CompileError 的 422 转换通道。"""


def calibrate_runs(
    commands: list[dict[str, Any]],
    records: list[dict[str, Any]],
    tolerance_ms: int,
) -> dict[str, Any]:
    """按通道估计实播时钟偏移，返回各通道偏移与一致 / 异常记录 id。

    不会修改传入的 ``commands`` / ``records``。

    :raises CompileError: 计划口令非法（重复 id、依赖环等，与 /compile 相同）。
    :raises CalibrateError: 容差为负、实播记录字段非法、记录 id 未出现在计划中、
        或记录通道与计划通道不一致时抛出，``code`` 可用于程序化区分错误类型。
    """
    if tolerance_ms < 0:
        raise CalibrateError(
            "INVALID_REQUEST",
            "tolerance_ms 必须是非负数",
            {"tolerance_ms": tolerance_ms},
        )

    # 1) 计划侧：复用 /compile 的锚点解析（拷贝一份，避免 setdefault 改动调用方数据；
    #    校准只使用时间线，跳过 O(n²) 的冲突对枚举）
    timeline = compile_commands(
        [dict(item) for item in commands], with_conflicts=False
    )["timeline"]
    plan_by_id = {entry["id"]: entry for entry in timeline}

    # 2) 实播记录字段级校验（与 /runs/reconcile 同一套规则）
    validate_records(records, CalibrateError)

    # 3) 校准语义：记录 id 必须出现在计划中，且通道与对应计划口令一致
    record_channel = {record["id"]: record["channel"] for record in records}
    unknown = sorted(rid for rid in record_channel if rid not in plan_by_id)
    if unknown:
        raise CalibrateError(
            "CALIBRATION_TARGET_NOT_FOUND",
            f"实播记录引用了计划中不存在的口令 id: {', '.join(unknown)}",
            {"ids": unknown},
        )
    mismatched = sorted(
        rid
        for rid, channel in record_channel.items()
        if channel != plan_by_id[rid]["channel"]
    )
    if mismatched:
        raise CalibrateError(
            "CHANNEL_MISMATCH",
            f"实播记录通道与计划口令通道不一致: {', '.join(mismatched)}",
            {
                "ids": mismatched,
                "mismatches": [
                    {
                        "id": rid,
                        "plan_channel": plan_by_id[rid]["channel"],
                        "record_channel": record_channel[rid],
                    }
                    for rid in mismatched
                ],
            },
        )

    # 4) 按通道分组，逐通道把每条记录换算成使其双端一致的整数偏移闭区间
    intervals_by_channel: dict[str, list[OffsetInterval]] = {}
    for record in records:
        plan = plan_by_id[record["id"]]
        start_dev = record["actual_start_ms"] - plan["start_ms"]
        end_dev = record["actual_end_ms"] - plan["end_ms"]
        # offset 使该记录双端一致 <=> |start_dev - offset| 与 |end_dev - offset|
        # 都不超过容差，即 offset 同时落在两个偏差各自的容差邻域内
        lo = max(start_dev, end_dev) - tolerance_ms
        hi = min(start_dev, end_dev) + tolerance_ms
        intervals_by_channel.setdefault(record["channel"], []).append(
            (lo, hi, record["id"])
        )

    # 5) 逐通道选最优偏移并划分一致 / 异常记录（通道按名升序，id 按字典序升序）
    channels: list[dict[str, Any]] = []
    for channel in sorted(intervals_by_channel):
        entries = intervals_by_channel[channel]
        offset = _best_offset([(lo, hi) for lo, hi, _ in entries])
        if offset is None:
            # 所有记录在任何整数偏移下都无法满足双端容差
            channels.append(
                {
                    "channel": channel,
                    "status": "uncalibratable",
                    "offset_ms": None,
                    "consistent_ids": [],
                    "abnormal_ids": sorted(rid for _, _, rid in entries),
                }
            )
            continue
        consistent = sorted(
            rid for lo, hi, rid in entries if lo <= offset <= hi
        )
        abnormal = sorted(
            rid for lo, hi, rid in entries if not lo <= offset <= hi
        )
        channels.append(
            {
                "channel": channel,
                "status": "calibrated",
                "offset_ms": offset,
                "consistent_ids": consistent,
                "abnormal_ids": abnormal,
            }
        )
    return {"channels": channels}


def _best_offset(intervals: list[tuple[int, int]]) -> int | None:
    """从各记录的「可一致」整数偏移闭区间中选出最优偏移。

    返回被最多区间覆盖的整数偏移；覆盖数并列时先取绝对值最小、再取数值较小者。
    所有区间均为空（任何偏移都无法让对应记录双端一致）时返回 ``None``。
    """
    events: list[tuple[int, int]] = []
    for lo, hi in intervals:
        if lo > hi:
            continue  # 该记录在任何偏移下都无法双端一致，不参与覆盖
        events.append((lo, 1))
        events.append((hi + 1, -1))
    if not events:
        return None

    events.sort()
    best_key: tuple[int, int, int] | None = None  # (-覆盖数, |偏移|, 偏移)
    best_offset: int | None = None
    coverage = 0
    idx = 0
    while idx < len(events):
        x = events[idx][0]
        while idx < len(events) and events[idx][0] == x:
            coverage += events[idx][1]
            idx += 1
        # 覆盖度在 [x, next_x) 内恒定；覆盖度为正时必有后继坐标
        # （全部事件的增量之和为 0，最后一个坐标之后覆盖度必归零）
        next_x = events[idx][0] if idx < len(events) else None
        if coverage <= 0 or next_x is None:
            continue
        # 段内按 (|偏移|, 偏移) 取最优点：含 0 取 0；全正取左端；全负取最靠近 0 的点
        if x <= 0 < next_x:
            candidate = 0
        elif x > 0:
            candidate = x
        else:
            candidate = next_x - 1
        key = (-coverage, abs(candidate), candidate)
        if best_key is None or key < best_key:
            best_key = key
            best_offset = candidate
    return best_offset

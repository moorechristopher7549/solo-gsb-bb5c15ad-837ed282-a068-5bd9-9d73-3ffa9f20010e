"""app.calibrate 的单元测试（仅依赖标准库，可直接 `python -m unittest discover`）。"""

from __future__ import annotations

import copy
import unittest

from app.calibrate import CalibrateError, calibrate_runs
from app.compiler import CompileError


def cmd(cid, channel, anchor="START", offset=0, duration=1000):
    return {
        "id": cid,
        "channel": channel,
        "anchor": anchor,
        "offset_ms": offset,
        "duration_ms": duration,
    }


def rec(rid, channel, start, end):
    return {
        "id": rid,
        "channel": channel,
        "actual_start_ms": start,
        "actual_end_ms": end,
    }


def calibrate(commands, records, tolerance=100):
    return calibrate_runs(commands, records, tolerance)


def channel_result(result, channel):
    for entry in result["channels"]:
        if entry["channel"] == channel:
            return entry
    raise AssertionError(f"通道 {channel} 不在响应中: {result}")


class BasicTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(calibrate([], []), {"channels": []})

    def test_single_exact_record_prefers_zero_offset(self):
        # 完全一致的记录给出的偏移区间包含 0，绝对值最小者胜出
        result = calibrate(
            [cmd("a", "main", duration=1000)],
            [rec("a", "main", 0, 1000)],
        )
        entry = channel_result(result, "main")
        self.assertEqual(entry["status"], "calibrated")
        self.assertEqual(entry["offset_ms"], 0)
        self.assertEqual(entry["consistent_ids"], ["a"])
        self.assertEqual(entry["abnormal_ids"], [])

    def test_uniform_shift_is_recovered(self):
        # a [0,1000)；b 锚定 a 结束时刻 -> [1000,1500)；实播整体晚 30ms，容差 0 时
        # 唯一能让记录双端一致的偏移就是 30
        result = calibrate(
            [
                cmd("a", "main", duration=1000),
                cmd("b", "main", anchor="a", offset=0, duration=500),
            ],
            [rec("a", "main", 30, 1030), rec("b", "main", 1030, 1530)],
            tolerance=0,
        )
        entry = channel_result(result, "main")
        self.assertEqual(entry["offset_ms"], 30)
        self.assertEqual(entry["consistent_ids"], ["a", "b"])
        self.assertEqual(entry["abnormal_ids"], [])

    def test_shift_within_tolerance_prefers_zero_offset(self):
        # 整体晚 30ms 但容差 50：偏移 0 已让两条记录双端达标，按绝对值最小裁决取 0
        result = calibrate(
            [cmd("a", "main", duration=1000)],
            [rec("a", "main", 30, 1030)],
            tolerance=50,
        )
        entry = channel_result(result, "main")
        self.assertEqual(entry["offset_ms"], 0)
        self.assertEqual(entry["consistent_ids"], ["a"])

    def test_channels_sorted_and_only_record_channels_present(self):
        # 计划中的 sub 通道没有实播记录，不出现在响应中；通道按名升序
        result = calibrate(
            [
                cmd("a", "main", duration=1000),
                cmd("b", "sub", duration=1000),
                cmd("c", "abc", duration=1000),
            ],
            [rec("a", "main", 0, 1000), rec("c", "abc", 5, 1005)],
        )
        self.assertEqual([e["channel"] for e in result["channels"]], ["abc", "main"])

    def test_multiple_channels_get_independent_offsets(self):
        # main 整体晚 50ms、sub 整体早 50ms，容差 30：各通道独立估计出 20 与 -20
        result = calibrate(
            [
                cmd("a", "main", duration=1000),
                cmd("b", "sub", offset=2000, duration=1000),
            ],
            [rec("a", "main", 50, 1050), rec("b", "sub", 1950, 2950)],
            tolerance=30,
        )
        self.assertEqual(channel_result(result, "main")["offset_ms"], 20)
        self.assertEqual(channel_result(result, "sub")["offset_ms"], -20)

    def test_does_not_mutate_input(self):
        # 故意省略 anchor / offset_ms（/compile 会做 setdefault），调用后入参不应被改动
        commands = [{"id": "a", "channel": "main", "duration_ms": 1000}]
        records = [rec("a", "main", 0, 1000)]
        snapshot = copy.deepcopy((commands, records))
        calibrate(commands, records)
        self.assertEqual((commands, records), snapshot)


class OffsetSelectionTests(unittest.TestCase):
    def test_majority_wins_and_outlier_is_abnormal(self):
        # a、b 整体晚 100ms，c 晚 900ms：多数派区间 [50,150] 覆盖数 2 胜出，
        # 段内绝对值最小的 50 成为估计偏移，c 记为异常
        result = calibrate(
            [
                cmd("a", "main", offset=0, duration=1000),
                cmd("b", "main", offset=2000, duration=1000),
                cmd("c", "main", offset=4000, duration=1000),
            ],
            [
                rec("a", "main", 100, 1100),
                rec("b", "main", 2100, 3100),
                rec("c", "main", 4900, 5900),
            ],
            tolerance=50,
        )
        entry = channel_result(result, "main")
        self.assertEqual(entry["offset_ms"], 50)
        self.assertEqual(entry["consistent_ids"], ["a", "b"])
        self.assertEqual(entry["abnormal_ids"], ["c"])

    def test_deviation_equal_to_tolerance_is_consistent(self):
        # 偏差恰好等于容差（“不超过”含等号）：偏移 0 下仍计为一致
        result = calibrate(
            [cmd("a", "main", duration=1000)],
            [rec("a", "main", 100, 1100)],
            tolerance=100,
        )
        entry = channel_result(result, "main")
        self.assertEqual(entry["offset_ms"], 0)
        self.assertEqual(entry["consistent_ids"], ["a"])

    def test_double_end_required(self):
        # b 的实播时长被拉长 300ms（超过 2 倍容差），任何偏移都无法让它双端一致
        result = calibrate(
            [
                cmd("a", "main", offset=0, duration=1000),
                cmd("b", "main", offset=2000, duration=1000),
            ],
            [rec("a", "main", 0, 1000), rec("b", "main", 2000, 3300)],
            tolerance=100,
        )
        entry = channel_result(result, "main")
        self.assertEqual(entry["offset_ms"], 0)
        self.assertEqual(entry["consistent_ids"], ["a"])
        self.assertEqual(entry["abnormal_ids"], ["b"])

    def test_stretched_record_can_still_calibrate_within_double_tolerance(self):
        # 实播时长比计划多 100ms（恰为 2 倍容差）：偏移 50 时两端偏差 ±50 均达标
        result = calibrate(
            [cmd("a", "main", duration=800)],
            [rec("a", "main", 0, 900)],
            tolerance=50,
        )
        entry = channel_result(result, "main")
        self.assertEqual(entry["status"], "calibrated")
        self.assertEqual(entry["offset_ms"], 50)
        self.assertEqual(entry["consistent_ids"], ["a"])

    def test_tie_prefers_smaller_absolute_offset(self):
        # 单条记录偏移区间为 [5, 15]：覆盖数都是 1，取绝对值最小的 5
        result = calibrate(
            [cmd("a", "main", duration=1000)],
            [rec("a", "main", 10, 1010)],
            tolerance=5,
        )
        self.assertEqual(channel_result(result, "main")["offset_ms"], 5)

    def test_tie_prefers_smaller_absolute_offset_negative_side(self):
        # 计划 [2000,3000)，实播早 10ms：偏移区间为 [-15, -5]，取最靠近 0 的 -5
        result = calibrate(
            [cmd("a", "main", offset=2000, duration=1000)],
            [rec("a", "main", 1990, 2990)],
            tolerance=5,
        )
        self.assertEqual(channel_result(result, "main")["offset_ms"], -5)

    def test_tie_same_absolute_prefers_smaller_value(self):
        # a 的区间为 [5, 15]、b 的区间为 [-15, -5]：候选 ±5 绝对值相同，取较小值 -5
        result = calibrate(
            [
                cmd("a", "main", offset=0, duration=1000),
                cmd("b", "main", offset=2000, duration=1000),
            ],
            [rec("a", "main", 10, 1010), rec("b", "main", 1990, 2990)],
            tolerance=5,
        )
        entry = channel_result(result, "main")
        self.assertEqual(entry["offset_ms"], -5)
        self.assertEqual(entry["consistent_ids"], ["b"])
        self.assertEqual(entry["abnormal_ids"], ["a"])

    def test_zero_offset_wins_inside_wide_interval(self):
        # 区间 [-100, 100] 内 0 的绝对值最小
        result = calibrate(
            [cmd("a", "main", duration=1000)],
            [rec("a", "main", 3, 1003)],
            tolerance=100,
        )
        self.assertEqual(channel_result(result, "main")["offset_ms"], 0)


class UncalibratableTests(unittest.TestCase):
    def test_single_record_stretched_beyond_double_tolerance(self):
        # 实播时长比计划多 300ms > 2 倍容差：任何偏移都无法双端一致
        result = calibrate(
            [cmd("a", "main", duration=1000)],
            [rec("a", "main", 0, 1300)],
            tolerance=100,
        )
        entry = channel_result(result, "main")
        self.assertEqual(entry["status"], "uncalibratable")
        self.assertIsNone(entry["offset_ms"])
        self.assertEqual(entry["consistent_ids"], [])
        self.assertEqual(entry["abnormal_ids"], ["a"])

    def test_uncalibratable_channel_alongside_calibrated_one(self):
        # main 整体晚 150ms（容差 100 -> 估计偏移 50）；sub 时长被拉长 500ms，无法校准
        result = calibrate(
            [cmd("a", "main", duration=1000), cmd("b", "sub", duration=1000)],
            [rec("a", "main", 150, 1150), rec("b", "sub", 0, 1500)],
            tolerance=100,
        )
        main = channel_result(result, "main")
        self.assertEqual(main["status"], "calibrated")
        self.assertEqual(main["offset_ms"], 50)
        sub = channel_result(result, "sub")
        self.assertEqual(sub["status"], "uncalibratable")
        self.assertIsNone(sub["offset_ms"])
        self.assertEqual(sub["abnormal_ids"], ["b"])


class ValidationTests(unittest.TestCase):
    def test_unknown_record_id(self):
        with self.assertRaises(CalibrateError) as ctx:
            calibrate(
                [cmd("a", "main")],
                [rec("a", "main", 0, 1000), rec("x", "main", 0, 1000)],
            )
        self.assertEqual(ctx.exception.code, "CALIBRATION_TARGET_NOT_FOUND")
        self.assertEqual(ctx.exception.details["ids"], ["x"])

    def test_channel_mismatch(self):
        with self.assertRaises(CalibrateError) as ctx:
            calibrate(
                [cmd("a", "main")],
                [rec("a", "sub", 0, 1000)],
            )
        self.assertEqual(ctx.exception.code, "CHANNEL_MISMATCH")
        self.assertEqual(ctx.exception.details["ids"], ["a"])
        self.assertEqual(
            ctx.exception.details["mismatches"],
            [{"id": "a", "plan_channel": "main", "record_channel": "sub"}],
        )

    def test_unknown_id_checked_before_channel_mismatch(self):
        # 记录 id 不在计划中时无从比较通道，优先报未知 id
        with self.assertRaises(CalibrateError) as ctx:
            calibrate([cmd("a", "main")], [rec("x", "sub", 0, 1000)])
        self.assertEqual(ctx.exception.code, "CALIBRATION_TARGET_NOT_FOUND")

    def test_negative_tolerance(self):
        with self.assertRaises(CalibrateError) as ctx:
            calibrate_runs([], [], -1)
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")

    def test_duplicate_record_id(self):
        with self.assertRaises(CalibrateError) as ctx:
            calibrate(
                [cmd("a", "main")],
                [rec("a", "main", 0, 1000), rec("a", "main", 0, 1000)],
            )
        self.assertEqual(ctx.exception.code, "DUPLICATE_ID")
        self.assertEqual(ctx.exception.details["duplicates"], ["a"])

    def test_empty_record_id(self):
        with self.assertRaises(CalibrateError) as ctx:
            calibrate([], [rec("", "main", 0, 100)])
        self.assertEqual(ctx.exception.code, "INVALID_ID")

    def test_empty_record_channel(self):
        with self.assertRaises(CalibrateError) as ctx:
            calibrate([], [rec("a", "", 0, 100)])
        self.assertEqual(ctx.exception.code, "INVALID_CHANNEL")

    def test_invalid_interval(self):
        for start, end in ((500, 400), (500, 500), (-1, 500)):
            with self.assertRaises(CalibrateError) as ctx:
                calibrate([], [rec("a", "main", start, end)])
            self.assertEqual(ctx.exception.code, "INVALID_INTERVAL")

    def test_plan_errors_propagate(self):
        # 计划侧沿用 /compile 的校验与错误码
        with self.assertRaises(CompileError) as ctx:
            calibrate(
                [cmd("a", "main", anchor="b"), cmd("b", "main", anchor="a")],
                [],
            )
        self.assertEqual(ctx.exception.code, "DEPENDENCY_CYCLE")

    def test_plan_checked_before_records(self):
        # 计划与实播同时非法时，先报计划侧错误
        with self.assertRaises(CompileError) as ctx:
            calibrate(
                [cmd("a", "main", duration=0)],
                [rec("a", "main", 500, 400)],
            )
        self.assertEqual(ctx.exception.code, "INVALID_DURATION")


if __name__ == "__main__":
    unittest.main()

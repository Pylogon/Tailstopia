"""Offline notification checks; never import or launch the game input adapter."""
import ast
import ctypes
from ctypes import wintypes
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "tailstopia_auto_reroll.py"


class RerollNotificationTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
        names = {"run_cli", "show_status_popup", "success_details", "format_stats", "target_met"}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.stop = threading.Event()
        self.fail_safe = type("FailSafeException", (Exception,), {})
        self.ns = {"stop_requested": self.stop, "DEBUG_DIR": Path("tailstopia_debug"),
                   "pyautogui": SimpleNamespace(FailSafeException=self.fail_safe),
                   "ctypes": ctypes, "wintypes": wintypes}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), self.ns)
        self.popup = Mock(side_effect=lambda *args: self.assertTrue(self.stop.is_set()))
        self.ns["show_status_popup"] = self.popup

    def run_main(self, callback):
        self.ns["main"] = callback
        with patch("builtins.print"):
            self.ns["run_cli"]()

    def test_success_popup_once_after_main_resources_close(self):
        closed = []
        def main():
            try:
                return "SUCCESS", "CHA=10 AGI=9 TOTAL=37"
            finally:
                closed.append(True)
        self.run_main(main)
        self.assertEqual(closed, [True])
        self.popup.assert_called_once_with("SUCCESS", "CHA=10 AGI=9 TOTAL=37")

    def test_error_keeps_exact_reason_and_absolute_debug_path(self):
        self.run_main(Mock(side_effect=ValueError("OCR ambiguous: CHA=None")))
        status, message = self.popup.call_args.args
        self.assertEqual(status, "ERROR")
        self.assertIn("ValueError: OCR ambiguous: CHA=None", message)
        self.assertIn(str(Path("tailstopia_debug").resolve()), message)

    def test_focus_loss_and_unchanged_stats_use_error_notification(self):
        for reason in ("Game lost focus", "Stats unchanged twice"):
            with self.subTest(reason=reason):
                self.stop.clear()
                self.popup.reset_mock()
                self.run_main(lambda: ("ERROR", reason))
                self.popup.assert_called_once_with("ERROR", reason)

    def test_ctrl_c_and_failsafe_are_user_stops(self):
        for error in (KeyboardInterrupt(), self.fail_safe()):
            with self.subTest(error=type(error).__name__):
                self.stop.clear()
                self.popup.reset_mock()
                self.run_main(Mock(side_effect=error))
                self.assertEqual(self.popup.call_args.args[0], "STOPPED")
                self.popup.assert_called_once()

    def test_f9_during_ocr_is_stop_instead_of_error(self):
        def main():
            self.stop.set()
            raise RuntimeError("F9 pressed during OCR")
        self.run_main(main)
        self.assertEqual(self.popup.call_args.args[0], "STOPPED")

    def test_f9_and_limit_notifications_forward_once(self):
        for reason in ("F9 pressed", "Safety limit reached"):
            with self.subTest(reason=reason):
                self.stop.clear()
                self.popup.reset_mock()
                self.run_main(lambda: ("STOPPED", reason))
                self.popup.assert_called_once_with("STOPPED", reason)

    def test_declining_initial_values_does_not_show_success_or_error(self):
        self.run_main(lambda: None)
        self.popup.assert_not_called()
        self.assertTrue(self.stop.is_set())

    def test_success_details_include_all_stats_total_and_no_create(self):
        values = {"CHA": 10, "AGI": 9, "INT": 6, "TAL": 6, "VIT": 6}
        detail = self.ns["success_details"](values)
        for name, value in values.items():
            self.assertIn(f"{name}={value}", detail)
        self.assertIn("TOTAL = 37", detail)
        self.assertIn("CREATE has not been clicked", detail)

    def test_exact_targets_and_sum_boundary_are_preserved(self):
        values = {"CHA": 10, "AGI": 9, "INT": 6, "TAL": 6, "VIT": 6}
        target = self.ns["target_met"]
        self.assertTrue(target(values))
        for changed in ({"VIT": 5}, {"CHA": 11}, {"CHA": 9}, {"AGI": 10}, {"AGI": 8}):
            self.assertFalse(target({**values, **changed}))
        self.assertTrue(target({**values, "VIT": 8}))

    def test_tk_unavailable_falls_back_to_topmost_native_popup(self):
        self.ns["user32"] = SimpleNamespace(MessageBoxW=Mock())
        tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "show_status_popup")
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), self.ns)
        with patch.dict("sys.modules", {"tkinter": None}), patch("builtins.print"):
            self.ns["show_status_popup"]("ERROR", "OCR failed")
        args = self.ns["user32"].MessageBoxW.call_args.args
        self.assertEqual(args[1], "OCR failed")
        self.assertTrue(args[3] & 0x00040000)  # MB_TOPMOST
        self.assertTrue(args[3] & 0x10)  # MB_ICONERROR


if __name__ == "__main__":
    unittest.main()

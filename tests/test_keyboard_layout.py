import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tailstopia_workflow.game_window import require_english_keyboard


class KeyboardLayoutTests(unittest.TestCase):
    def api(self,layout,thread=321):
        return SimpleNamespace(GetWindowThreadProcessId=Mock(return_value=thread),
                               GetKeyboardLayout=Mock(return_value=layout))

    def test_accepts_english_layout_variants_from_game_thread(self):
        for language in (0x0409,0x0809,0x0C09):
            with self.subTest(language=language):
                api = self.api((language << 16)|language)
                self.assertEqual(require_english_keyboard(api,123),language)
                api.GetWindowThreadProcessId.assert_called_once_with(123,None)
                api.GetKeyboardLayout.assert_called_once_with(321)

    def test_thai_layout_blocks_with_manual_switch_instructions(self):
        with self.assertRaisesRegex(RuntimeError,"0x041E.*Switch to English"):
            require_english_keyboard(self.api(0x041E041E),123)

    def test_unavailable_layout_is_not_assumed_english(self):
        with self.assertRaisesRegex(RuntimeError,"Cannot read game keyboard layout"):
            require_english_keyboard(self.api(0),123)

    def test_invalid_thread_blocks_before_layout_lookup(self):
        api = self.api(0x0409,thread=0)
        with self.assertRaisesRegex(RuntimeError,"Cannot read game keyboard thread"):
            require_english_keyboard(api,123)
        api.GetKeyboardLayout.assert_not_called()

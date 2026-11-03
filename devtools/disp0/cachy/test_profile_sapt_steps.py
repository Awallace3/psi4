"""Stdlib-only timer/selection tests; no native imports or GPU work."""
import unittest

from profile_sapt_steps import GIB, admit_device, case_policy, parse_timers


HEADER = "Module                               User System Wall Calls\n"
FLAT = (
    "SAPT(DFT) Energy : 20.000u 1.000s 7.000w 1 calls\n"
    "SAPT(DFT):Dimer SCF : 9.000u 0.500s 2.000w 1 calls\n"
)


class DriverProfileTests(unittest.TestCase):
    def test_flat_only_uses_wall_not_user_or_duplicate_tree(self):
        text = HEADER+FLAT+"-"*86+"\n"+FLAT+"*"*86+"\n"
        report = parse_timers(text)
        self.assertEqual(report["driver_energy"]["wall_seconds"], 7)
        self.assertEqual(report["driver_steps"]["SAPT(DFT):Dimer SCF"]["wall_seconds"], 2)
        self.assertIsNone(report["driver_steps"]["SAPT(DFT):exch"])

    def test_empty_initial_epoch_is_not_an_extra_calculation(self):
        report = parse_timers(HEADER+"-"*86+"\n"+"*"*86+"\n"+HEADER+FLAT)
        self.assertEqual(report["nonempty_epochs"], 1)

    def test_appended_independent_epochs_reject(self):
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            parse_timers(HEADER+FLAT+"*"*86+"\n"+HEADER+FLAT)

    def test_duplicate_flat_label_and_missing_driver_reject(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_timers(HEADER+FLAT+FLAT)
        with self.assertRaisesRegex(ValueError, "completed SAPT"):
            parse_timers(HEADER+"Other : 1.000u 0.000s 1.000w 1 calls\n")

    def test_bounded_cases_preserve_source_and_proof(self):
        base = {"source_pin": "frozen", "build_proof": "unchanged", "threads": 99}
        water = case_policy(base, "water", "/private/water")
        peptide = case_policy(base, "peptide", "/private/peptide")
        self.assertEqual(water["source_pin"], "frozen")
        self.assertEqual(peptide["build_proof"], "unchanged")
        self.assertEqual(base["threads"], 99)
        self.assertEqual(peptide["threads"], 4)
        self.assertEqual(water["device_memory_limit_bytes"], GIB)
        self.assertEqual(peptide["device_memory_limit_bytes"], 20*GIB)
        with self.assertRaises(ValueError):
            case_policy(base, "protein157", "/private/protein157")
        with self.assertRaises(ValueError):
            case_policy(base, "nanotube", "/private/nanotube")

    def test_desktop_headroom_and_busy_guards(self):
        config = {"device_memory_limit_bytes": 20*GIB}
        admit_device({"free_MiB": 22517, "utilization_percent": 0}, config)
        with self.assertRaises(MemoryError):
            admit_device({"free_MiB": 20000, "utilization_percent": 0}, config)
        with self.assertRaisesRegex(RuntimeError, "busy"):
            admit_device({"free_MiB": 22517, "utilization_percent": 50}, config)


if __name__ == "__main__":
    unittest.main()

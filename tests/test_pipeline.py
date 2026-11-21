"""Test không cần camera, không cần file .pt:  .venv\\Scripts\\python -m unittest discover tests"""

import os
import socket
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cau_hinh  # noqa: E402
from cau_hinh import ClassCfg  # noqa: E402
from detector import find_model_file  # noqa: E402
from distance import estimate_distance  # noqa: E402
from postprocess import Detection, drop_riders, plausible, suppress_cross_class  # noqa: E402
from sender import DistanceSender, format_message  # noqa: E402
from tracker import Tracker  # noqa: E402

CAR = ClassCfg(key="o_to", model="models/o_to.pt", real_height_m=1.50, real_width_m=1.73)
ORDER = ("xe_may", "nguoi_di_bo", "o_to")


class ModelSetupTests(unittest.TestCase):
    def test_default_config_only_needs_the_three_model_files(self):
        # chỉ cần chép xe_may.pt / nguoi_di_bo.pt / o_to.pt vào là chạy: không phải sửa tên lớp hay chiều cao
        self.assertEqual([c.key for c in cau_hinh.CLASSES], ["xe_may", "nguoi_di_bo", "o_to"])
        for c in cau_hinh.CLASSES:
            self.assertEqual(c.model, f"models/{c.key}.pt")
            self.assertEqual(c.names, ())
            self.assertTrue(c.real_height_m, c.key)

    def test_model_file_is_found_in_models_folder_or_project_folder(self):
        with tempfile.TemporaryDirectory() as base:
            self.assertIsNone(find_model_file("models/o_to.pt", base))
            open(os.path.join(base, "o_to.pt"), "wb").close()              # thả nhầm ra thư mục ngoài vẫn nhận
            self.assertEqual(find_model_file("models/o_to.pt", base), os.path.join(base, "o_to.pt"))
            os.mkdir(os.path.join(base, "models"))
            open(os.path.join(base, "models", "o_to.pt"), "wb").close()    # có trong models/ thì ưu tiên models/
            self.assertEqual(find_model_file("models/o_to.pt", base), os.path.join(base, "models", "o_to.pt"))


class DistanceTests(unittest.TestCase):
    def test_same_formula_as_old_code(self):
        d = estimate_distance((100, 200, 160, 300), CAR, focal_px=1452, frame_w=1280, frame_h=720)
        self.assertAlmostEqual(d.value, 1452 * 1.50 / 100)          # (FOCAL_LENGTH * REAL_HEIGHT) / h_px
        self.assertFalse(d.truncated)

    def test_box_cut_by_image_edge_takes_the_shorter_estimate(self):
        # vật quá gần, hộp chạm đáy ảnh -> chiều cao trên ảnh bị cắt -> khoảng cách theo chiều cao bị ƯỚC LỚN
        d = estimate_distance((400, 520, 800, 720), CAR, 1452, 1280, 720)
        self.assertTrue(d.truncated)
        self.assertAlmostEqual(d.value, min(1452 * 1.50 / 200, 1452 * 1.73 / 400))


class PostprocessTests(unittest.TestCase):
    def test_same_object_seen_by_two_models_is_kept_once(self):
        car = Detection("o_to", (100, 100, 200, 200), 0.9, "o_to")
        bike = Detection("xe_may", (104, 102, 203, 198), 0.6, "xe_may")
        self.assertEqual(suppress_cross_class([car, bike], min_iou=0.7), [car])

    def test_two_neighbouring_objects_of_one_class_are_not_merged(self):
        a = Detection("o_to", (100, 100, 200, 200), 0.9, "")
        b = Detection("o_to", (190, 100, 290, 200), 0.8, "")
        self.assertEqual(len(suppress_cross_class([a, b], min_iou=0.7)), 2)

    def test_person_sitting_on_motorbike_is_not_a_pedestrian(self):
        bike = Detection("xe_may", (100, 150, 220, 300), 0.8, "")
        rider = Detection("nguoi_di_bo", (120, 60, 200, 250), 0.7, "")
        walker = Detection("nguoi_di_bo", (400, 60, 460, 300), 0.7, "")
        out = drop_riders([bike, rider, walker], pedestrian="nguoi_di_bo", vehicle="xe_may", overlap=0.5)
        self.assertEqual(out, [bike, walker])

    def test_tiny_box_is_rejected(self):
        self.assertFalse(plausible(Detection("o_to", (10, 10, 14, 16), 0.9, ""), CAR, min_height_px=10))
        self.assertTrue(plausible(Detection("o_to", (10, 10, 40, 40), 0.9, ""), CAR, min_height_px=10))


def feed(tr, t, items):
    return tr.update(t, [(Detection(slot, box, 0.9, ""), dist) for slot, box, dist in items])


class TrackerTests(unittest.TestCase):
    def tracker(self):
        return Tracker(match_iou=0.3, confirm_frames=3, max_missed_s=0.3, smooth_frames=5)

    def test_single_frame_false_detection_is_never_reported(self):
        tr = self.tracker()
        feed(tr, 0.0, [("o_to", (100, 100, 200, 200), 1.7)])
        for k in range(1, 10):
            feed(tr, k / 30, [])
        self.assertEqual(tr.nearest(), {})

    def test_two_cars_keep_their_own_ids_and_distances(self):
        tr = self.tracker()
        for k in range(10):
            feed(tr, k / 30, [("o_to", (100 + 2 * k, 100, 200 + 2 * k, 200), 1.70),
                              ("o_to", (700 - 2 * k, 300, 760 - 2 * k, 360), 2.90)])
        tracks = tr.confirmed()
        self.assertEqual(sorted(round(x.distance, 2) for x in tracks), [1.70, 2.90])
        self.assertEqual(len({x.id for x in tracks}), 2)
        self.assertAlmostEqual(tr.nearest()["o_to"], 1.70)

    def test_short_dropout_is_bridged_long_one_is_not(self):
        tr = self.tracker()
        for k in range(5):
            feed(tr, k / 30, [("o_to", (100, 100, 200, 200), 1.5)])
        feed(tr, 5 / 30 + 0.2, [])
        self.assertIn("o_to", tr.nearest())
        feed(tr, 5 / 30 + 0.5, [])
        self.assertEqual(tr.nearest(), {})

    def test_one_frame_distance_spike_is_filtered(self):
        tr = self.tracker()
        for k, d in enumerate((1.50, 1.50, 1.50, 9.0, 1.50)):
            feed(tr, k / 30, [("o_to", (100, 100, 200, 200), d)])
        self.assertAlmostEqual(tr.nearest()["o_to"], 1.50)


class MessageTests(unittest.TestCase):
    def test_exact_format(self):
        self.assertEqual(format_message({"xe_may": 1.234, "o_to": 0.5}, ORDER), "@distance:1.23;-1;0.50;;\r\n")

    def test_nothing_in_frame_sends_minus_one(self):
        self.assertEqual(format_message({}, ORDER), "@distance:-1;-1;-1;;\r\n")


class SenderTests(unittest.TestCase):
    def test_stale_or_missing_data_sends_failsafe(self):
        now = [0.0]
        s = DistanceSender(ORDER, udp_target=None, tcp_bind=None, rate_hz=20, stale_s=0.5, failsafe_m=0.0,
                           clock=lambda: now[0], start=False)
        self.assertEqual(s.current_message(), "@distance:0.00;0.00;0.00;;\r\n")     # chưa có frame nào
        s.publish({"o_to": 1.5})
        self.assertEqual(s.current_message(), "@distance:-1;-1;1.50;;\r\n")
        now[0] = 0.6                                                                  # camera/nhận diện bị treo
        self.assertEqual(s.current_message(), "@distance:0.00;0.00;0.00;;\r\n")

    def test_tcp_client_and_udp_receiver_get_a_continuous_stream(self):
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind(("127.0.0.1", 0))
        udp.settimeout(2.0)
        s = DistanceSender(ORDER, udp_target=("127.0.0.1", udp.getsockname()[1]), tcp_bind=("127.0.0.1", 0),
                           rate_hz=50, stale_s=5.0, failsafe_m=0.0)
        try:
            s.publish({"o_to": 1.5})
            cli = socket.create_connection(("127.0.0.1", s.tcp_port), timeout=2.0)
            buf, deadline = b"", time.time() + 2.0
            while buf.count(b"\r\n") < 5 and time.time() < deadline:
                buf += cli.recv(4096)
            lines = buf.split(b"\r\n")[:5]
            self.assertTrue(all(x == b"@distance:-1;-1;1.50;;" for x in lines), lines)
            got = [udp.recv(1024) for _ in range(6)]          # gói đầu có thể là bản tin an toàn trước publish()
            self.assertEqual(got[-1], b"@distance:-1;-1;1.50;;\r\n")
            cli.close()
        finally:
            s.close()
            udp.close()


if __name__ == "__main__":
    unittest.main()

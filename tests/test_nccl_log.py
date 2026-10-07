from instrument.nccl_log import parse_dir, parse_text

# Excerpt of the real Kaggle 2xT4 NCCL INFO log (results/raw/smoke/nccl_98784e75b20e_195.log).
EXCERPT = """\
98784e75b20e:195:195 [0] NCCL INFO NCCL version 2.31.2+cuda13.3
98784e75b20e:195:195 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 1
98784e75b20e:195:195 [0] NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct
98784e75b20e:195:195 [0] NCCL INFO Channel 01 : 0[0] -> 1[1] via SHM/direct
"""


def test_parse_text():
    r = parse_text(EXCERPT)
    assert r["runtime_version"] == "2.31.2+cuda13.3"
    assert r["transports"] == ["SHM/direct"]
    assert r["p2p_check"].startswith("Check P2P Type isAllDirectP2p 0")
    assert len(r["via_lines"]) == 2


def test_parse_dir(tmp_path):
    (tmp_path / "nccl_h_1.log").write_text(EXCERPT, encoding="utf-8")
    (tmp_path / "nccl_h_2.log").write_text(EXCERPT.replace("0[0] -> 1[1]", "1[1] -> 0[0]"), encoding="utf-8")
    r = parse_dir(tmp_path)
    assert len(r["log_files"]) == 2 and len(r["via_lines"]) == 4


def test_empty_dir(tmp_path):
    r = parse_dir(tmp_path)
    assert r["transports"] == [] and r["runtime_version"] is None

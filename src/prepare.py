import io
import os
import random
import re
import sys
import xml.etree.ElementTree
from pathlib import Path

from zntrack import NodeConfig, nodify


@nodify(
    params={"split": 0.20, "seed": 20170428},
    outs=Path("data", "prepared"),
    deps=[Path("data", "data.xml"), Path("src", "prepare.py")],
)
def prepare(cfg: NodeConfig):
    output_train = cfg.outs / "train.tsv"
    output_test = cfg.outs / "test.tsv"

    def process_posts(fd_in, fd_out_train, fd_out_test, target_tag):
        num = 1
        for line in fd_in:
            try:
                fd_out = (
                    fd_out_train if random.random() > cfg.params.split else fd_out_test
                )
                attr = xml.etree.ElementTree.fromstring(line).attrib

                pid = attr.get("Id", "")
                label = 1 if target_tag in attr.get("Tags", "") else 0
                title = re.sub(r"\s+", " ", attr.get("Title", "")).strip()
                body = re.sub(r"\s+", " ", attr.get("Body", "")).strip()
                text = title + " " + body

                fd_out.write("{}\t{}\t{}\n".format(pid, label, text))

                num += 1
            except Exception as ex:
                sys.stderr.write(f"Skipping the broken line {num}: {ex}\n")

    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    with io.open(cfg.deps[0], encoding="utf8") as fd_in:
        with io.open(output_train, "w", encoding="utf8") as fd_out_train:
            with io.open(output_test, "w", encoding="utf8") as fd_out_test:
                process_posts(fd_in, fd_out_train, fd_out_test, "<python>")

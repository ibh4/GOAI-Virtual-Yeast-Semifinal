"""命令 1: 外部数据 / embedding 构建。

最终模型 L2b (C2-r256 + FC-PCC loss) **不使用任何外部数据或
embedding**——无菌株基因组、化合物结构、蛋白序列、通路、PPI 或
其他公开知识进入模型输入。

本脚本为显式 no-op, 退出码 0, 仅输出声明以满足复现流程的第一步。
"""
import sys


def main():
    print("[build_embeddings] 最终模型不使用外部数据或 embedding — "
          "无需执行, 本脚本为显式 no-op。")
    print("[build_embeddings] 外部数据来源声明见 "
          "external_data/source_manifest.json (entries = none)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Official LIBERO Dataset Archive

本页只管理 LIBERO 官方 demonstration HDF5 归档。它是可长期复用的源数据，不从属于
FastWAM、CodeWAM 或任一训练格式。

## 范围与固定版本

下载源固定为官方 LIBERO 下载脚本使用的 Hugging Face dataset repository:

```text
repository: yifengzhu-hf/LIBERO-datasets
revision:   f13aa24a3da8c43c7225569f28c562979fa0e35a
files:      132 total = 130 HDF5 + 2 metadata
license:    CC BY 4.0
```

官方 benchmark 语义上包含 `libero_spatial`、`libero_object`、`libero_goal` 和
`libero_100`。当前官方仓库把 `libero_100` 物理拆成 `libero_90` 与 `libero_10`，所以完整
落盘目录是:

```text
libero_spatial/  10 tasks
libero_object/   10 tasks
libero_goal/     10 tasks
libero_90/       90 tasks
libero_10/       10 tasks
```

不要把它与 `libero_*_no_noops_lerobot` 混在一起。后者是供特定训练管线使用的转换产物，
可以由官方归档派生，但不能替代官方归档。

## 开发集群一键下载

在包含本仓库的开发机 shell 中执行:

```bash
cd /path/to/CodeWAM
export DATA_ROOT=/mnt/gpu11_200T/users/dingxibo/datasets
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/gpu11_200T/users/dingxibo/cache/huggingface

bash scripts/download_libero_official.sh all
```

脚本不创建虚拟环境、不安装依赖，也不写入 home 根目录。它使用当前默认环境中的 `hf`，
默认四并发，并在传输中断时断点续传和自动重试。可调参数:

```bash
export HF_MAX_WORKERS=4
export MAX_ATTEMPTS=30
export RETRY_DELAY_SECONDS=30
export VERIFY_WORKERS=4
```

长任务可放入 `tmux`:

```bash
mkdir -p "${DATA_ROOT}/logs"
tmux new-session -d -s libero_official \
  "cd '${PWD}' && bash scripts/download_libero_official.sh all 2>&1 | \
  tee '${DATA_ROOT}/logs/libero-official-f13aa24.log'"
tmux attach -t libero_official
```

## 分阶段执行

同一入口支持四种模式:

```bash
bash scripts/download_libero_official.sh manifest
bash scripts/download_libero_official.sh download
bash scripts/download_libero_official.sh verify
bash scripts/download_libero_official.sh all
```

最终布局:

```text
${DATA_ROOT}/
├── libero/official/                         # 官方 HDF5 与仓库元数据
└── manifests/libero/official-f13aa24/
    ├── source.json                          # 来源、revision、规模与目录语义
    ├── expected_files.json                  # 官方远端文件清单
    ├── expected_hdf5.sha256                 # 130 个官方 LFS SHA256
    └── verification.json                    # 本地验收结果
```

`verify` 同时检查五个目录的任务数、130 个 HDF5 文件头、额外/缺失文件及全部 SHA256。
只有 `verification.json` 的 `status` 为 `ok`，才把该归档视为可用于后续转换和实验。

## 当前集群归档

2026-07-23 已在开发集群完成固定版本归档:

```text
root:            /mnt/gpu11_200T/users/dingxibo/datasets/libero/official
manifest:        /mnt/gpu11_200T/users/dingxibo/datasets/manifests/libero/official-f13aa24
HDF5 files:      130
HDF5 bytes:      100442942572
verification:    status=ok, sha256_checked=130
incomplete:      0
owner:           scut:scut
```

实际使用前以共享盘中的 `verification.json` 为准，不能仅凭本页的历史状态假定数据仍完整。

## 后续派生边界

官方归档保持只读。FastWAM/LeRobot、robomimic adapter、视频导出或 CodeWAM latent cache
都写入独立目录，并在各自 manifest 中记录:

```text
source revision + source SHA256 + converter commit + conversion config
```

这样可以从任一派生数据追溯回官方 HDF5，同时允许多个项目共享同一份源数据。

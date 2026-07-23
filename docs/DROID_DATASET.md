# Official DROID RLDS Archive

本页管理 DROID 官方 RLDS 数据。它是真机机器人示范底库，不绑定 FastWAM、CodeWAM 或某一种
训练格式。

## 版本选择

DROID 官方将 RLDS 定位为策略训练和高效 dataloading 格式，将 Raw 数据定位为需要高清、
立体、深度或原始相机信息的场景。CodeWAM 当前采用:

```text
full training archive: gs://gresearch/robotics/droid/1.0.1
debug archive:         gs://gresearch/robotics/droid_100/1.0.0
not downloaded:        droid_raw
license:               CC BY 4.0
```

当前 bucket 仍保留旧的 `droid/1.0.0`。它与 `1.0.1` 是两个完整 RLDS 版本，不应重复下载。
固定的 `1.0.1` 包含 2048 个 TFRecord shards，共 2050 个对象、1,865,994,705,042 字节。
DROID-100 包含 31 个 shards，共 33 个对象、2,192,615,094 字节。

## 一键下载

在开发集群仓库中执行:

```bash
cd /mnt/gpu11_200T/users/dingxibo/CodeWAM
export DATA_ROOT=/mnt/gpu11_200T/users/dingxibo/datasets

bash scripts/download_droid_official.sh all
```

脚本使用默认环境中的 `gsutil`，不创建虚拟环境。它会:

```text
snapshot GCS object generations and checksums
-> download and verify DROID-100
-> resumably download full RLDS 1.0.1
-> verify exact paths, sizes, metadata and every GCS MD5
```

长任务建议放入 `tmux`:

```bash
mkdir -p "${DATA_ROOT}/logs"
tmux new-session -d -s droid_official \
  "cd '/mnt/gpu11_200T/users/dingxibo/CodeWAM' && \
  DATA_ROOT='${DATA_ROOT}' bash scripts/download_droid_official.sh all 2>&1 | \
  tee '${DATA_ROOT}/logs/droid-rlds-1.0.1.log'"
tmux attach -t droid_official
```

默认使用一个进程和 16 个并发线程。可调参数:

```bash
export GSUTIL_PROCESSES=1
export GSUTIL_THREADS=16
export MAX_ATTEMPTS=100
export RETRY_DELAY_SECONDS=60
export VERIFY_WORKERS=4
```

分阶段入口:

```bash
bash scripts/download_droid_official.sh manifest
bash scripts/download_droid_official.sh debug
bash scripts/download_droid_official.sh full
bash scripts/download_droid_official.sh verify
```

## 最终布局

```text
${DATA_ROOT}/
├── droid/1.0.1/
├── droid_100/1.0.0/
└── manifests/droid/
    ├── rlds-1.0.1/
    │   ├── source.json
    │   ├── objects.json
    │   ├── expected_files.md5
    │   └── verification.json
    └── droid-100-rlds-1.0.0/
        ├── source.json
        ├── objects.json
        ├── expected_files.md5
        └── verification.json
```

任何 LeRobot、视频文件、Wan pooled latent 或过滤子集都必须写入独立派生目录，并记录源 GCS
generation、MD5、转换代码 commit 与配置。

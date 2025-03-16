[（雑多なメモ）報酬が連続値の場合のオフ方策評価の精度検証](https://zenn.dev/shuto51/articles/9c6a1372f957ce)内の結果を出力するためのリポジトリ。

# Experiments

| id  | desc                                                   |
|----|--------------------------------------------------------|
| 1  | 1.2.2-1.2.4 dm-ips-dr に近い形で OPE やってみる。IPS 推定量のばらつき確認 |
| 2  | 1 を連続値にしてみる                                   |
| 3  | 2 で beta = 1                                        |
| 4  | 2 で beta = 0.1                                      |
| 5  | 4 でpi_0 推定                                        |
| 6  | 4 でpi 推定                                         |
| 7  | 4 でpi_0, pi 推定                                   |
| 8  | 4 で only positive reward (未完成)                   |

今後試したいこと。オンライン実験、信頼区間、only positive reard, ほぼ0 reward、SNIPS、DRなど。

# Build Docker image

```bash
docker build -t ope-continuous-reward .
```

# Run Docker container

```bash
docker run \
    -it \
    --rm \
    -v $(pwd):/work \
    --entrypoint python \
    ope-continuous-reward \
    exps/<xxx>.py
```

# References

- https://github.com/ghmagazine/cfml_book

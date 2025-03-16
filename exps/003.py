import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.utils import check_random_state
from tqdm import tqdm

plt.style.use("ggplot")
y_label_dict = {"se": "左図：平均二乗誤差", "bias": "中図：二乗バイアス", "variance": "右図：バリアンス"}

import numpy as np

# import open bandit pipeline (obp)
from obp.dataset import (  # SyntheticBanditDatasetWithActionEmbeds as SyntheticBanditDataset,
    SyntheticBanditDataset,
    linear_reward_function,
    logistic_polynomial_reward_function,
)
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import OffPolicyEvaluation
from pandas import DataFrame
from scipy.stats import rankdata


def eps_greedy_policy(
    expected_reward: np.ndarray,
    k: int = 1,
    eps: float = 0.1,
) -> np.ndarray:
    """epsilon-greedy法により方策を定義する."""
    is_topk = rankdata(-expected_reward, axis=1) <= k
    pi = ((1.0 - eps) / k) * is_topk
    pi += eps / expected_reward.shape[1]
    pi /= pi.sum(1)[:, np.newaxis]

    return pi[:, :, np.newaxis]


def aggregate_simulation_results(
    estimated_policy_value_list: list,
    policy_value: float,
    experiment_config_name: str,
    experiment_config_value: int,
) -> DataFrame:
    """各推定量の推定値から平均二乗誤差や二乗バイアス、バリアンスなどの実験結果を集計する."""
    result_df = (
        DataFrame(DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "value"})
    )
    result_df[experiment_config_name] = experiment_config_value
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = 0
    result_df["variance"] = 0
    result_df["true_value"] = policy_value
    sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
    for est_ in sample_mean["est"]:
        estimates = result_df.loc[result_df["est"] == est_, "value"].values
        mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["est"] == est_, "bias"] = (policy_value - mean_estimates) ** 2
        result_df.loc[result_df["est"] == est_, "variance"] = (estimates - mean_estimates) ** 2

    return result_df


def aggregate_simulation_results_lam(
    estimated_policy_value_list: list,
    policy_value: float,
    experiment_config_name: str,
    experiment_config_value: int,
) -> DataFrame:
    """各推定量の推定値から平均二乗誤差や二乗バイアス、バリアンスなどの実験結果を集計する."""
    result_df = (
        DataFrame(DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "lam", 0: "value"})
    )
    result_df[experiment_config_name] = experiment_config_value
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = 0
    result_df["variance"] = 0
    result_df["true_value"] = policy_value
    sample_mean = DataFrame(result_df.groupby(["lam"]).mean().value).reset_index()
    for est_ in sample_mean["lam"]:
        estimates = result_df.loc[result_df["lam"] == est_, "value"].values
        mean_estimates = sample_mean.loc[sample_mean["lam"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["lam"] == est_, "bias"] = (policy_value - mean_estimates) ** 2
        result_df.loc[result_df["lam"] == est_, "variance"] = (estimates - mean_estimates) ** 2

    return result_df


## シミュレーション設定
num_runs = 100  # シミュレーションの繰り返し回数
dim_context = 10  # 特徴量xの次元
n_actions = 2  # 行動数, |A|
beta = 1  # データ収集方策のパラメータ
test_data_size = 100000  # 評価方策の真の性能を近似するためのテストデータのサイズ
random_state = 12345
random_ = check_random_state(random_state)
num_data_list = [250, 500, 1000, 2000, 4000, 8000]  # データ収集方策が収集したログデータのサイズ
num_data_list = [8000]  # データ収集方策が収集したログデータのサイズ

result_df_list = []
for num_data in num_data_list:
    ## 人工データ生成クラス
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        action_context=random_.normal(size=(n_actions, 10)),
        beta=beta,
        reward_type="continuous",
        reward_function=linear_reward_function,
        reward_std=0,
        random_state=random_state,
    )

    ## 評価方策の真の性能(policy value)を近似するためのテストデータ
    test_data = dataset.obtain_batch_bandit_feedback(n_rounds=test_data_size)

    ## 評価方策の真の性能(policy value)を近似
    policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=test_data["expected_reward"],
        action_dist=eps_greedy_policy(test_data["expected_reward"]),
    )

    estimated_policy_value_list = []
    for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
        ## データ収集方策が形成する分布に従いログデータを生成
        offline_logged_data = dataset.obtain_batch_bandit_feedback(n_rounds=num_data)
        print(offline_logged_data["reward"].mean())

        ## ログデータ上における評価方策の行動選択確率を計算
        pi = eps_greedy_policy(offline_logged_data["expected_reward"])

        ## ログデータを用いてオフ方策評価を実行する
        ope = OffPolicyEvaluation(
            bandit_feedback=offline_logged_data,
            ope_estimators=[
                IPS(estimator_name="IPS"),
            ],
        )
        estimated_policy_values = ope.estimate_policy_values(
            action_dist=pi,  # \pi(a|x)
        )
        estimated_policy_value_list.append(estimated_policy_values)

    ## シミュレーション結果を集計する
    result_df_list.append(
        aggregate_simulation_results(
            estimated_policy_value_list,
            policy_value,
            "num_data",
            num_data,
        )
    )
result_df = pd.concat(result_df_list).reset_index(level=0)
result_df = pd.concat([result_df, result_df.tail(1)]).reset_index(drop=True)
# make last row as a true value
result_df.loc[result_df.index[-1], "est"] = "true"
result_df.loc[result_df.index[-1], "value"] = policy_value
result_df.loc[result_df.index[-1], "num_data"] = num_data_list[-1]
result_df.loc[result_df.index[-1], "se"] = 0
result_df.loc[result_df.index[-1], "bias"] = 0
result_df.loc[result_df.index[-1], "variance"] = 0
result_df.loc[result_df.index[-1], "true_value"] = policy_value
print(result_df)

sns.catplot(x="est", y="value", data=result_df)
save_dir = Path("artifacts")
save_dir.mkdir(exist_ok=True)
plt.savefig(save_dir / f"{Path(__file__).stem}.png")

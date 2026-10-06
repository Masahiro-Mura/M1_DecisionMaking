# 主論文との比較用グラフ

- fig7_learning_comparison.png / .pdf：Fig.7と同じ2行3列構成。左は報酬、中央は成功率、右は学習全体の平均。現在CBFG-PPOのみ各1 seed。PPO-R・PPO・DDPGはN/A。1 seedのためseed間SD・帯は描かない。
- fig8_cbfg_ppo_speed5_seed0.png / .pdf：Fig.8と同じEV・SVのx座標、y座標、速度の3段構成。10 m/sも別ファイルに保存。
- Fig.9はPPOの実測ログが必要。現在は未学習なので未生成。PPOのログをmanifestに追加すると同じ3段構成で生成する。

Fig.8は既存評価で選ばれた代表エピソードを使う。主論文の凝縮ケースとSV数・配置・軌跡が一致したものではない。SV番号はログの車両IDを保持し、近傍車だけへの恣意的な選び直しや時間軸の移動をしない。時間0の行がないので0.2 sから実測値を表示する。

成功率は直近3000 stepに完了したエピソードの成功割合を各完了時点で計算し、幅10の移動平均を表示する。報酬の線は判読のため幅10の移動平均を使う（本文には指定がない描画上の仮定）。棒の報酬集計には平滑化前の全エピソード報酬を使う。右の棒は各runの全学習平均を先に計算し、seedを等重みで平均する。成功率の棒は表示平滑化前の率を使う。完了時点でのサンプリング頻度、seedの曲線間の線形補間、SDの分母（ddof=0）は描画上の仮定。途中で学習予算に達した未完了エピソードは除外する。

学習データは各150000 stepのv4 run。評価軌跡は既存bestPolicyのもの。論文の平均的な学習returnのモデルを選ぶ手順とは異なる。論文値・未実施の比較手法・10 seedの統計は補っていない。描画変更に伴う再学習は行っていない。

再生成（プロジェクト直下）：
`python -m main.mainPaperFigures`

比較手法やseed追加：input_manifest.jsonを編集したコピーに各runを追加し、
`python -m main.mainPaperFigures --manifest PATH_TO_MANIFEST`

各runのdirectory/logsにtrainingLog.csvを置く。simulationLog.csvがあればFig.8/9も生成する。元CSVのSHA256、集計値、描画規約はplot_manifest.jsonで確認できる。

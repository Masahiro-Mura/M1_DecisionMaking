# [43]・[44]・[37]を照合した追加修正

追加修正、各速度1 seed・150000 stepの再学習、最良・終了時モデル各50エピソードの評価を完了しました。最良モデルは最低速度5 m/sで72%、10 m/sで28%、終了時は68%、22%です。55項目のテストは全件成功しました。論文性能の完全再現には未到達です。

追加資料から、実演とpolicyの行動表現の整合性、離散時間CBFの安全条件を確認しました。教師変換と重みの座標系を修正し、安全性の前提条件を診断に追加しました。ただし追加資料も、主論文固有のMPC重み、介入閾値、CBF教師行動の逆変換法を補完していません。新しい変換法と重みの座標系は仮定であり、著者実装と一致すると判定したものではありません。

## 引用論文から確認できたこと

|資料|該当箇所|今回の扱い|
|---|---|---|
|[43] Vecerik et al., arXiv:1707.08817v2|Sec.3（PDF2～3ページ）、Sec.4.1（PDF5ページ）|DDPGfDは実演とagentの同じ行動空間を使う。実演の優先replay・n-step return等はDDPG用であり、PPOへ追加しない|
|[44] Rajeswaran et al., arXiv:1709.10087v2|Sec.IV-C、Eq.(5)～(6)（PDF5～6ページ）|BCは実演行動の対数尤度を使い、DAPGは実演重みを減衰する。主論文Eq.(28)の指数重み、Eq.(31)の二乗誤差を優先し、尤度・減衰へ置換しない|
|[37] Agrawal and Sreenath|Proposition4、Definition4、Remark3（PDF4ページ）|条件はΔh+γh≥0、0<γ≤1、初期h≥0。1−γという解釈を確認。入力制限がある場合の常時実行可能性は保証されない|

主論文は[43][44]をLfD/LfIの枠組みとして引用していますが、そこにCBFG-PPOの加速度・操舵角から参照速度・headingへの変換手順はありません。[44]のλ0=0.1、λ1=0.95を追加すると、違う学習則になります。

## 主論文の最低速度条件の読み違いを修正

主論文III-D（PDF9ページ）は、5 m/s・10 m/sを初期速度だけでなく走行中に維持する最低速度として記載しています。前回の実装・報告では初期速度として扱い、走行中にそれ以下へ減速できました。この見落としを訂正します。最低速度条件は主論文にあるため、追加文献がないことで生じた未記載条件ではありません。時間切れに関わる重要なシナリオ差です。

- 正規化行動を物理速度へ戻す範囲の下限を、そのシナリオの5 m/sまたは10 m/sへ変更しました。上限20 m/sとheading範囲は引き続き未記載の仮定です。
- MPCでは全予測stepに v_j = v_0 + dt * sum(acceleration_i) >= v_min という線形制約を加えました。主論文の最低速度条件を離散運動モデルへ反映させる実装上の選択です。
- CBF補正後も同じ条件を守るため、加速度下限に (v_min - v_current)/dt を反映しました。CBFの解がなくなる場合に下限を無視して適用することはありません。
- 教師行動の変換・逆変換も同じ速度範囲を使います。model・evaluation manifestに最低速度を保存しました。
- 適用後の車両速度を強制的に書き換える処理はありません。予測モデルと実際の加速度更新を揃えます。

下限付きの変換と逆変換、教師・agentの変換範囲、全horizonのMPC速度、CBF補正後の速度、入力制限で到達不可能な状態の拒否、実環境への最小参照速度の連続適用をテストし、55項目全件成功しました。

教師・重みだけを変更したv3の途中runは停止し、最低速度も修正したv4を新規学習し、150000 stepと評価を完了しました。v3の途中checkpointとstdoutは診断用として保持し、完成した再学習結果として扱いません。v2-v4の比較にはこの交通条件の訂正も含まれるため、教師変換だけの効果を判定する比較ではありません。

## 教師行動の修正

従来はCBF補正後の1 step先の実現速度・headingを教師参照そのものとしていました。元の目標速度18 m/s、実速度5 m/s、名目加速度1 m/s²という診断条件で、操舵だけの補正を与えると教師速度は5.2 m/sになります。微小補正でもこの置換が起きる一方、補正ゼロ時は元の行動を返すため、ゼロ付近で不連続です。CBF介入の効果と、目標にまだ到達していない追従誤差を混同していました。

同じ状態・運動モデルによる安全制御と名目制御の差だけを元の参照へ加える方法に変更しました。

```text
x_nom_next = F(x, u_MPC)
x_safe_next = F(x, u_MPC + correction)
delta_ref = x_safe_next[v, heading] - x_nom_next[v, heading]
teacher_ref = original_policy_ref + delta_ref
teacher_action = normalize_and_clip(teacher_ref)
```

ゼロ補正で元の行動を保ち、操舵だけの補正で速度参照を変えません。加速度補正2 m/s²、dt=0.2 sなら元の参照速度へ0.4 m/sを加えます。teacher_speed_diagnostic.pngはこの診断であり、学習性能の図ではありません。

これはMPCのhorizon全体の逆変換ではなく、次回MPCが同じ補正済み制御を出力する保証はありません。引用論文から確定した式ではなく、主論文Eq.(26)の加法的補正と同じ行動空間での学習を両立させるための修正仮説です。安全filterは引き続き毎step適用します。

## 模倣損失の重み

誤差は正規化した速度・headingで計算していましたが、重みには物理的な加速度・操舵角のnormを用いていました。主論文がどの空間でΔaを定義するかは曖昧です。新しい仮定では、保存した教師と元のpolicy行動の差を同じ正規化座標で計算します。

```text
rollout_delta = stored_teacher_action - stored_original_action
weight = 1 + exp(norm(rollout_delta))
BC = mean(weight * squared_norm(current_policy_action - stored_teacher_action))
```

重みは現在のpolicyとの誤差から毎epoch作り直さず、収集時の差に固定します。指数形式はEq.(28)のままです。2次元行動が[-1,1]内なので重みは2～約17.92になり、恣意的な指数clipは不要です。ただし、著者が正規化座標で重みを計算した証拠はありません。

安全controllerは物理的な制御を使います。介入flagは従来の物理制御norm>1e-3を維持しています。閾値の単位・尺度も未記載で、重みの座標系とは別の未確定条件です。

## CBFの診断

gammaが有限値かつ0<gamma≤1でない場合は初期化で拒否します。制約の1−gammaは維持しました。Eq.(20)と[37]Proposition4から導出でき、主論文Eq.(25)の車両CBFの1+gammaとは不整合です。

SafeControlResultと環境infoへ、現在のbarrier最小値と初期安全集合条件を満たすかを追加しました。solverの残差が通ることと、初期安全集合に属することを分けて確認できます。各速度200 seedの初期化では違反0件、barrier最小値2 mでした。この範囲では初期条件違反が主因とは考えにくいです。

初期h<0からのforward invarianceは主張しません。外側道路境界の既存の解釈と制約は変更していません。[37]のCLF用slackをCBFへ流用していません。定理は離散時刻・採用モデルに対する条件であり、予測誤差、時刻間の接触、楕円近似と物理車体の差まで保証するものではありません。

## 検証・実験条件

55項目のテストが全件成功しました。既存39項目に加え、教師変換の連続性、操舵だけの介入、加速度補正のdt、bounds、重み付き二乗誤差の値と勾配、重みの範囲、gamma定義域、CBF減衰式、安全集合外のsolver判定、未適用制御の教師除外を確認しています。

```powershell
python -m unittest diagnostics.testAdditionalReferences diagnostics.testReferencePapers diagnostics.testSingleLanePaper diagnostics.testImplementationFixes -q
```

再学習は最低速度5 m/s・10 m/s（初期速度も同値）、seed0、各150000 step、40epoch、3072 buffer、batch128です。最良policyは独立した固定10シナリオで選択し、最良・終了時policyをseed10000～10049の各50エピソード、周辺車の追加速度変動±1 m/sで評価します。

教師変換と重みを同時に変更しているため、どちらが性能差の原因かを単独で特定する実験ではありません。最低速度条件も訂正しているため、教師変換だけの効果は判定できません。各1 seedなので統計的な改善を断定しません。著者実装固有のパラメータが不明という問題は残ります。

## 保存先

プロジェクト: C:/Users/mura/Simulation/Research activities/M1_DecisionMaking/CBFG_PPO_Python

- 変更前コード: diagnostics/additionalReferenceAudit/before。
- 前回の1車線run: results/runs/speed5_seed0 と speed10_seed0。
- 新run: results/runs/reference_v4_speed5_seed0 と reference_v4_speed10_seed0。
- 同梱: additional_changes.patch、experiment_manifest.json、teacher_and_initial_state_diagnostics.json。
- modelの実装versionとsource hashを更新し、前回policyを今回の学習済みモデルとして流用しません。

## 参照PDF

- [43]: C:/Users/mura/Downloads/1707.08817v2 (1).pdf
- [44]: C:/Users/mura/Downloads/1709.10087v2 (1).pdf
- [37]: C:/Users/mura/Downloads/07593c39a72088663c03dafeef7865743d8a (1).pdf
- 主論文: C:/Users/mura/OneDrive - keio.jp/高橋研/研究室/M1_研究室/論文サーベイ/合流制御/① Control Barrier Function-Guided Deep Reinforcement Learning for Decision-Making of Autonomous Vehicle at On-Ramp Merging.pdf

## 完了した再学習・評価の結果

|速度 (m/s)|モデル|前回v2|今回v4|今回の終了理由|
|---|---|---|---|---|
|5|最良|66%|36/50 (72%)|{'success': 36, 'cbf_infeasible': 13, 'collision': 1}|
|5|終了時|70%|34/50 (68%)|{'success': 34, 'cbf_infeasible': 15, 'collision': 1}|
|10|最良|32%|14/50 (28%)|{'success': 14, 'cbf_infeasible': 35, 'collision': 1}|
|10|終了時|2%|11/50 (22%)|{'success': 11, 'cbf_infeasible': 39}|

両速度とも150000 stepを完了しました。各モデル50エピソードの評価を完了し、学習曲線・評価図・代表動画を再生成しました。成功時の平均合流位置、全終了理由はresults.jsonにあります。これらの数値は論文性能の完全再現や、複数seedでの有意な改善を証明するものではありません。

指定されたプロジェクトのresults/figures・results/videosは今回の5 m/s最良policyの結果へ更新しました。以前の既定resultsはdiagnostics/additionalReferenceAudit/previousDefaultResultsに退避し、前回の個別runも保持しています。本報告と同じフォルダのspeed5・speed10に各速度の図・動画・CSVをコピーしています。動画は代表エピソードであり、全試験の成功を示すものではありません。

## 残る性能差の解釈

10 m/sの終了時policyは前回2%から22%へ上がりましたが、最良policyは前回32%から28%となり、全条件で改善したわけではありません。5 m/sの最良policyは66%から72%となりましたが、各1 seed・50試験であり有意な改善の証明ではありません。

今回の10 m/s評価は最良policyで35/50、終了時policyで39/50がcbf_infeasibleでした。前回の終了時policyでは39/50がtime_limitでした。今回は時間切れが0件となりましたが、失敗を解消したという意味ではありません。最低速度の維持により減速・待機の選択肢が狭くなり、CBFの実行可能性との両立がより強く要求されます。最低速度の数値は主論文にありますが、著者がどの層でどう維持したかは未記載です。

cbf_infeasibleはsolver・回復候補で実行可能な制御を得られなかった分類で、解が数学的に存在しない証明ではありません。道路CBFの解釈、初期交通分布、教師の逆変換、MPC重み・horizon、成功判定の数値、policyのbounded Gaussian表現と決定的行動の扱いが残る再現上の条件です。主論文が認めるCBFの実行可能性の限界だけで全性能差を説明していません。

4モデルのsource hashと最低速度metadataの整合、および代表動画の全記録stepでEV速度と参照速度が下限以上であることを確認しました。動画の始点・中間・終点、比較図、車両y座標図を目視確認しました。これらは全試験の無衝突証明ではありません。追加の引用論文の依頼は現時点ではありません。

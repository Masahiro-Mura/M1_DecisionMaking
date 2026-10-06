# 引用論文[46]・[47]の照合結果

## [46] Treiber et al., Congested Traffic States in Empirical Observations and Microscopic Simulations

PDF4ページEq.(2)は接近速度を後続車速度−前車速度と定義しています。現在の符号は正しいため維持しました。Eq.(3)はnet (bumper-to-bumper) distanceを定義しており、車両中心間距離をそのまま用いる従来コードは原典に対応していません。周辺車両のIDM入力を、車両中心間距離から両車の半車体長を差し引いた距離に修正しました。同じ5 m車両なら中心間20 mはnet gap15 m、中心間12 mはnet gap7 mです。

PDF5～6ページEq.(6)～(7)のdesired gapには、従来コードのmax(d0, desired_gap)はありません。この未記載の下限を削除しました。原典には追加項s1 sqrt(v/v0)がありますが、同ページ本文でs1=0を採用しています。主論文Eq.(33)もこの項を含まないため、s1を新しく導入する必要はありません。

IDMの数値パラメータは主論文Table IIの値を優先し、原典の別実験の数値に置き換えていません。gapが0以下の場合の1e-3 mの除算保護は、衝突・重なり状態で有限値を保つ数値上の仮定として残しています。原典の連続時間での無衝突性を、0.2 s離散シミュレーションで保証するとは述べません。

## [47] Schulman et al., Proximal Policy Optimization Algorithms

PDF3ページEq.(7)のclipped surrogate objectiveは、現在のppo_actor_lossと対応します。gradient descentでは符号を反転しています。0.2はSection3の例およびSection6.2の設定に明記されています。

PDF5ページEq.(11)～(12)には、delta_t + gamma lambda delta_(t+1) + ...のtruncated GAEが記載されています。PDF10ページAppendix A Table3のlambda=0.95と現在の設定は対応します。episodeのreset境界を越えて次episodeのGAEを加算しない既存の修正も維持しました。

clip幅0.2、lambda0.95の選択には原典の根拠があります。ただし、主論文の著者が同じ値を使ったとは断定できません。原典の64 unit/tanh、lr=3e-4等は別実験の値であり、主論文Fig.6の256 unit/ReLU、TableIIIのlr=6e-4等を置き換えていません。

PPO自体は補正済み加速度・操舵角から速度・headingへのCBF教師行動の逆変換を記載していません。これについてはユーザーの指示に従い、現在の1step先状態を教師とする仮定を明記して進めます。

## 計算方法の変更

MPCの速度・headingの追従costについて、Eq.(16)の同じ状態更新に対するadjoint gradientを与えました。x/yはこのcostと速度・headingの更新に影響しないため、objective計算中だけ省略しています。最終的なMPC予測軌道は元の4状態rolloutで作ります。12組のランダム状態・20個のcontrol変数で元costとの一致と中心差分との一致を検証しました。SLSQP、horizon10、bounds、許容誤差、40epoch、150000stepは維持しました。

CBFは同じ各scalar不等式を1個のvector callbackにまとめ、objectiveに正確な微分2*correctionを与えました。元の不等式、safe set、制約許容誤差は変えていません。短い計測の200stepではprofilingを含む計算時間が約10.2秒から約2.1秒に短縮しました。これは計算方法の計測であり、学習性能の比較ではありません。

## 評価条件

今回はユーザーが指定した5 m/s・10 m/sの各1 seed、150000stepです。Fig.7の10 seed平均・標準偏差や他手法との比較は、今回の結果にあるとは扱いません。

主論文IV-Bは試験時に速度の追加変動を述べますが幅は未記載です。今回の50episode試験では既存8～12 m/sの周辺初期速度に、独立uniform ±1 m/sを加える仮定を設定しました。evaluationManifest.jsonに幅・seed・policyの出所を保存します。幅0に設定すれば追加変動なしとの比較もできます。

原著コード・補足資料はユーザーがお持ちでないため、CBF教師・道路CBFの解釈・成功target等を著者実装と同一であるとは判定できません。主論文内の式や学習stepの不整合も記録します。

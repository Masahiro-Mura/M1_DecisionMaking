指摘した不具合と再現上の曖昧さに対する修正を実装しました。14個のproduction fileを変更し、ディレクトリ構造を維持しています。既存bestPolicy.pt/finalPolicy.ptは上書きしていません。15万step・40epoch・buffer3072も維持しています。

論文が数値や変換を指定していない部分を、論文そのものの規定として扱っていません。新しい解釈・数値・failure処理にはコード内で`IMPLEMENTATION ASSUMPTION`を明記しました。特にCBF教師の逆変換と道路安全集合には下記の再現上の限定があります。

|修正ファイル|修正関数|修正前の問題|修正内容|論文根拠|実装仮定|テスト結果|
|---|---|---|---|---|---|---|
|main/mainLearning.py|train, evaluate_policy, select_best_policy|未更新policyをtraining中の確率的成功率でbestに選び、最後のpartial更新を評価しなかった|更新後だけ独立した固定10seedのdeterministic評価で選ぶ。成功率→平均returnの順で比較し、同点は最新更新を採用。partial更新も対象。selection logを分離|Algorithm 1のpolicy更新。bestの選び方は論文未指定|あり：評価seed/episode数/ranking|初期policy選択禁止、劣るscoreは置換しない、同点の最新更新を採用、実学習partial更新のbest/final一致がPASS|
|rl/advantage.py, rl/rolloutBuffer.py, rl/ppoAgent.py, main/mainLearning.py|compute_gae, clear/add, store_transition/_prepare_advantages, train|time-limitで次episodeのGAEが混入|terminatedはbootstrap mask、terminated OR truncatedはrecursion maskとして別に保存・伝播|主論文はadvantage estimator未指定。引用[47]に基づく既存GAEのepisode処理を修正|GAE自体は既存の仮定。境界修正は実装の整合性|truncationでbootstrapを残してrecursion遮断、terminationでbootstrap遮断、連続trajectoryの再帰維持がPASS|
|rl/cbfGuidedLoss.py, environment/actionAdapter.py, rl/ppoAgent.py|build_demonstration_action, reference_to_action, store_transition/update|速度/heading referenceに加速度/steering補正を直接加算していた|feasibleで適用されたsafe controlからモデル1step先の速度/headingを計算し、adapterの逆変換で教師actionへ。非介入時は元actionを維持。拒否された制御はActor学習対象から除外|Eq.(14)〜(16)のモデル、Eq.(27)〜(31)の安全行動の模倣|あり：1stepで実現したreferenceをlocal teacherとする。有限horizon MPCの厳密な逆変換ではなく、そのreferenceの将来安全性を保証しない。毎stepのCBFを維持|物理単位・モデルとの数値一致、zero補正のaction保存、必要なstate/control未提供時の明示エラー、拒否入力の除外がPASS|
|mpc/mpcCost.py|trajectory_cost|N+1番目のstateにも追加tracking costを課していた|Eq.(17)通りk=0,...,N−1のみ。末端penaltyを削除|Eq.(17)、PDF5ページ|追加仮定なし。出力抽出Cの寸法解釈は既存のまま|末端stateだけ大きくしてもcostが変わらない。0/±reference、nonzero heading、飽和warm start、HighwayEnv step一致がPASS|
|cbf/safeController.py|correct, _build_constraints|Eq.(25)の文字通りの実装という出典説明が誤り。solverの失敗出力を適用し得た|Eq.(20)と整合する1−gammaを維持し、Eq.(25)の1+gammaとの不一致を説明。nominal feasibleならzero補正を直接返す。失敗時はbounded候補から可行性を回復し同じ制約で再最適化。残差を独立に検証|Eq.(20)のsafe-set invariance、Eq.(25)のminimum-norm correction|あり：印刷不一致の解決と数値的feasibility recovery。constraint slackは導入しない。solver statusだけで安全とは判定しない|安全nominalの補正0、unsafe時の介入、人工solver失敗からのfeasible recoveryがPASS|
|cbf/roadCBF.py, environment/mergeEnvironment.py, cbf/safeController.py|RoadCBFContext, left/right_road_barrier, _road_cbf_context, _build_constraints|車線ごとの安全集合・permissionで制約が切り替わり、外縁で無効化され得る。過去と現在のlane frameを混ぜる可能性があった|現在位置・deviation・tangentを同時に再設定。連結する走行可能lane intervalから道路外縁を構成。現在と1step先のcorridor intersectionを使ってramp消滅時の人工的なbarrier jumpを防ぐ。物理外縁では両側を常時検証し、next barrier>=0も要求|Eq.(19)〜(24)の非負な道路安全集合。HighwayEnv geometryは[45]|あり：printed Eq.(23)/(24)とactivationの曖昧さを、parallel merging/downstreamの物理外縁で解釈する。lcf/rcfの観測自体は変更せず、物理外縁をpermissionから独立させる。前進時のx到達範囲を使う現在シナリオ用のlocal approximation。論文の文字通りの式とは区別|lane ID切替で連続、混在frame除去、ramp終端検出、本線のsegment切替直進、外縁でflagsがFalseでも制約が有効、manual mergeがPASS|
|environment/mergeEnvironment.py, environment/termination.py, rl/ppoAgent.py|step, TerminationResult.failure, store_transition/update|可行性を確認できない入力を適用し、教師にし得た|MPC失敗/CBF可行解未検出でworldを進めず吸収的な失敗終了。control_applied=False、applied_control=Noneを返し、ActorのPPO/BCから除外。criticと既存failure rewardは使用|Algorithm 1の安全入力を適用する意図。failure時の具体的処理は未記載|あり：数値solver failure時のepisode終了。cbf_infeasibleは数学的に不可能と証明した意味ではなく、可行解未検出も含む|MPC/CBF拒否でego/SV位置不変、applied control無し、Actor教師にしないことがPASS|
|environment/termination.py, environment/mergeEnvironment.py, parameters/environmentParameters.py|check_merge_completed, check_success, evaluate_termination, reset/step|合流直後に即success終了し、その後の安定走行を見なかった|合流の候補判定と最終successを分離。連続main-lane安定stepを数え、target到達も確認。collision/departureのfailureを優先|III-Dの停止条件、IV-Aの合流後failure無し、IV-Bのtarget到達|あり：中心距離1mは既存仮定、追加の確認時間2s・target x=450mは論文未記載の設定値。450mはroad end460mの10m手前|時間不足・target未到達ではsuccessにしない、collision優先、逸脱/resetでcounter clear、manual feedbackでtarget成功がPASS|
|rl/ppoAgent.py, main/mainLearning.py, main/mainSimulation.py|save/load, train, simulate|policyに学習・環境の版情報がなく、未学習/古い環境のpolicyを通常評価に使えた|更新回数、実学習step・speed/seed/budget、source SHA-256を保存。通常simulationは学習更新とsource互換性を必須にする。resumeも互換性を確認。初速scenarioの違いも検出。diagnosticは旧policyを読み込み可能|再現性のための実装情報。論文数式とは独立|checkpoint schema/互換性判定は実装規約|未更新・不一致metadataの拒否、修正後の短期学習checkpointのload成功がPASS|
|main/mainSimulation.py, diagnostics/testMPCReference.py, diagnostics/testRoadCBFRollout.py|run_episode/write_trace_csv, run_case/cost_components, get_lane_diagnostics|拒否candidateを適用済みと誤認し得る。旧診断がlocal lane contextを再構成し、実際の新道路CBFと異なる値を表示する|適用フラグ・実入力を記録。simulationでは拒否時のcontrolをNaNにする。診断ではcandidateとappliedを区別し、実environmentのroad contextを使う。cost内訳もEq.(17)のstate範囲に追従|実データフローを検証する診断|診断処理であり、policy/rewardの変更なし|cost合計assert、適用nominal+correction一致、CSVによる適用失敗の区別を確認|
|parameters/rlParameters.py|BEST_POLICY_EVALUATION_*|bestの独立評価条件が未定義だった|10 episode、seed base1000000を設定に集約|checkpoint選択は論文未指定|あり。150000/40/3072は変更せず維持|設定値と元policyのSHA-256が維持されていることを確認|

回帰テストは`python -m unittest diagnostics.testImplementationFixes -v`で22件すべてPASS。テスト内の`isolated_checks`はさらに21パターンをassertします。実学習smokeは8step・40epoch、partial bufferの更新とbest/final保存・load・独立評価を実行しています。テストでは出力先を一時directoryへ、評価episode数だけ1へ切り替え、既存resultsを保護しています。通常学習の10episode評価とは区別してください。

`python -m compileall -q rl mpc cbf environment main parameters diagnostics`も成功しました。

旧finalPolicyの確認：

```powershell
python diagnostics/testMPCReference.py --model results/models/finalPolicy.pt --output diagnostics/fixes/rollout_final
```

旧policyは新しい教師・道路・成功条件で再学習されていません。A〜DのCSVとsummary.jsonを保存しています。Aは常時steering飽和が解消している旧finalPolicyの確認用で、学習後の性能保証ではありません。CBF可行解が見つからないstepは全caseで拒否終了します。nominalがfeasibleなstepの補正は0です。

交通を除いた手動feedback referenceによるintegration testでは、rampから本線中心へ進入してx>=450へ到達し、道路逸脱無しでsuccessとなりました。これはMPC/CBF/road/terminationの接続が動く証拠であり、RLが学習によって合流を獲得した証拠ではありません。15万stepの再学習・複数seedの性能評価はまだ行っていません。

再学習は旧checkpointをresumeせず、例えば速度5m/sで以下を実行してください。

```powershell
python main/mainLearning.py --speed 5 --total-steps 150000 --seed 0
python main/mainSimulation.py --speed 5 --model results/models/bestPolicy.pt
```

通常の学習entry pointはresultsのpolicy/logを保存・更新します。今回の作業では既存modelを上書きしていません。旧bestは未学習のため通常simulationで明示エラーになります。旧trained finalもsource metadataが無いため通常simulation/resumeでは拒否されますが、diagnostics/testMPCReference.pyで原因調査に利用できます。

変更前sourceは`before/`、production差分は`productionChanges.patch`、変更SHA-256一覧は`sourceChanges.json`、テスト結果は`regressionTests.log`に保存しました。

# Plan de test — AI-TestGuard

*Document généré par `scripts/make_test_plan.py` à partir de `docs/traceability.json` et `app/config.py`.*

## 1. Périmètre

Système sous test : chargement et validation des données, prétraitement sans état, pipeline scikit-learn
(imputation, `log1p(Amount)`, mise à l'échelle, modèle), réglage et seuil de décision, sauvegarde/rechargement
avec empreinte SHA-256, prédiction (`predict_fraud`, gestionnaire de requête), évaluation, non-régression,
robustesse, surveillance, sécurité opérationnelle et intégration continue.

## 2. Stratégie

Pyramide de tests : unitaires et propriétés (Hypothesis) → intégration (bout en bout, fuites, sauvegarde) →
évaluation statistique du modèle (métriques, intervalles de confiance, graines, partition temporelle) →
robustesse et non-régression → performance et sécurité → CI. Les tests rapides s'exécutent à chaque Pull Request ;
les tests lents (`-m slow`) et ceux qui exigent le jeu Kaggle (`-m full_data`) sont lancés manuellement ou sur calendrier.

## 3. Exigences, risques et traçabilité

| ID | Exigence | Type de test | Résultat attendu | Risque maîtrisé | Priorité | Tests liés / statut |
|---|---|---|---|---|---|---|
| T01 | Les colonnes nécessaires sont présentes | Données | Aucune colonne indispensable manquante | Prédictions décalées ou plantage si une colonne manque, est dupliquée ou permutée | Critique | 6 tests / 11 exéc. — ✔ |
| T02 | Les données respectent les types attendus | Unitaire | Types conformes au contrat | Coercition silencieuse (« 12,5 » devenu NaN), calculs erronés, plantage tardif | Critique | 5 tests / 9 exéc. — ✔ |
| T03 | Le prétraitement ne produit pas de valeurs non finies | Unitaire | Aucune valeur NaN ou infinie en sortie | NaN ou infinis propagés dans le modèle : probabilités invalides ou plantage | Critique | 6 tests / 6 exéc. — ✔ |
| T04 | Le modèle produit une probabilité valide | Fonctionnel ML | Probabilité comprise entre 0 et 1 | Décisions incohérentes, seuil inapplicable, erreurs en aval | Critique | 5 tests / 9 exéc. — ✔ |
| T05 | Les performances sont évaluées sur des données indépendantes | ML | Jeu de test non utilisé pour le réglage | Surestimation optimiste des performances (fuite de données) | Critique | 4 tests / 4 exéc. — ✔ |
| T06 | Les résultats sont reproductibles | Non-régression | Écart compatible avec la tolérance définie | Résultats invérifiables ; régression indétectable ; audit impossible | Majeure | 5 tests / 7 exéc. — ✔ |
| T07 | Les tests s'exécutent sur GitHub | CI | Workflow réussi ou échec correctement signalé | Défauts fusionnés sans contrôle ; échec masqué par la CI | Majeure | 4 tests / 5 exéc. — ✔ |
| T08 | Valeurs manquantes, infinies ou cible invalide : rejet explicite ou correction documentée | Unitaire / Données | Rejet avec message clair, ou imputation tracée et plafonnée (≤ 5 % par colonne) | Prédictions silencieusement fausses ; plantage en production ; données corrompues en amont non détectées | Critique | 9 tests / 13 exéc. — ✔ |
| T09 | Aucune fuite de données : transformations apprises sur l'entraînement seul, partitions disjointes, seuil réglé sur la validation | Intégration | Statistiques du scaler = celles de l'entraînement ; aucune ligne de validation/test vue au fit | Évaluation trop optimiste ; décision de mise en production fondée sur des chiffres faux | Critique | 5 tests / 5 exéc. — ✔ |
| T10 | Le seuil de décision est choisi sur la validation sous contrainte de taux de faux positifs | ML | Seuil maximisant le rappel avec FPR ≤ cible ; règle « p ≥ seuil » inclusive | Trop de fausses alertes (friction client) ou de fraudes manquées (pertes) | Critique | 5 tests / 11 exéc. — ✔ |
| T11 | Le modèle satisfait les critères d'acceptation métier (rappel, FPR, précision, PR-AUC, ROC-AUC) | ML | Chaque critère CA01 à CA05 évalué avec intervalle d'incertitude ; verdict mécanique | Mise en pilote d'un modèle inefficace ou intrusif | Critique | 8 tests / 9 exéc. — ✔ |
| T12 | Le pipeline sauvegardé puis rechargé produit les mêmes prédictions, avec contrôle d'intégrité SHA-256 | Intégration | Écart nul (≤ 1e-12) ; fichier altéré ou sans empreinte refusé avant désérialisation | Artefact corrompu ou altéré ; exécution de code arbitraire via pickle ; écart entre modèle testé et modèle déployé | Critique | 4 tests / 4 exéc. — ✔ |
| T13 | Les dimensions sont cohérentes de l'entrée à la sortie | Intégration | 29 variables transformées ; autant de probabilités que de lignes ; ordre des colonnes sans effet | Décalage de colonnes : prédictions erronées sans erreur visible | Majeure | 4 tests / 4 exéc. — ✔ |
| T14 | Le modèle reste stable sous perturbations réalistes (sensibilité bornée et mesurée) | Robustesse | Variation des probabilités sous la borne théorique ; croissance monotone avec l'amplitude ; taux de changement de classe mesuré | Décisions instables ; contournement par de petites modifications ; fausses alertes erratiques | Majeure | 6 tests / 6 exéc. — ✔ |
| T15 | Montants extrêmes, valeurs manquantes et décalage de distribution sont détectés (alerte ou rejet) | Robustesse / Monitoring | Comportement explicite par scénario : rejeter, traiter ou alerter | Dérive non détectée ; dégradation silencieuse ; scores incohérents sur des entrées atypiques | Majeure | 6 tests / 6 exéc. — ✔ |
| T16 | Aucune régression par rapport à la référence figée (tolérances définies, référence jamais recalculée seule) | Non-régression | Décision ACCEPTER / INVESTIGUER / REJETER motivée ; référence absente = échec | Dégradation introduite par une modification de code sans que personne ne le voie | Critique | 8 tests / 8 exéc. — ✔ |
| T17 | La latence P95, le débit, la taille et la mémoire respectent les budgets | Performance | Latence P95 ≤ 20 ms (service) / 50 ms (requête) ; débit ≥ 1 000 tx/s ; modèle ≤ 10 Mo | Dépassement du délai d'autorisation de paiement ; indisponibilité sous charge | Majeure | 4 tests / 4 exéc. — ✔ |
| T18 | Les entrées de l'API sont validées strictement (champs manquants ou supplémentaires, types, taille, constantes non finies) | Sécurité | Codes 400 / 413 / 422 explicites ; aucune erreur interne exposée | Injection, déni de service, comportement indéfini sur entrée malformée | Critique | 9 tests / 20 exéc. — ✔ |
| T19 | Aucun secret ni donnée sensible dans le dépôt Git et dans les journaux | Sécurité | 0 secret détecté ; journaux sans valeur d'entrée ; données et modèles ignorés par Git | Fuite d'identifiants ou de données de transaction ; publication involontaire du jeu de données | Critique | 7 tests / 7 exéc. — ✔ |
| T20 | La qualité du code est contrôlée automatiquement (Ruff, couverture, moindre privilège) | CI | Ruff sans erreur ; couverture ≥ 85 % ; permissions en lecture seule | Dette technique ; défauts non détectés ; compromission de la chaîne CI | Majeure | 4 tests / 4 exéc. — ✔ |
| T21 | Les métriques sont correctes sur données déséquilibrées (l'accuracy seule est écartée, cas limites traités) | ML | Précision, rappel, F1, FPR, FNR, ROC-AUC, PR-AUC conformes aux calculs de référence ; indéfini = NaN | Conclusions erronées : un classifieur inutile paraît excellent (accuracy > 99,8 %) | Critique | 8 tests / 15 exéc. — ✔ |
| T22 | La surveillance détecte la dérive des données et déclenche des alertes | Monitoring | PSI > 0,25 = alerte ; fenêtre stable = aucune alerte ; taux d'alertes et entrées invalides suivis | Dégradation silencieuse après déploiement (dérive de données ou de concept) | Majeure | 7 tests / 7 exéc. — ✔ |
| T23 | Les écarts de performance par segment sont mesurés avec leurs incertitudes ; les limites d'équité sont documentées | Équité | Intervalles de Wilson par segment ; segments à faible effectif signalés ; aucun attribut démographique inventé | Conclusion d'équité abusive à partir de données qui ne le permettent pas | Majeure | 3 tests / 3 exéc. — ✔ |

## 4. Cas de test par exigence

- **T01** : `tests/test_preprocessing.py::test_validate_accepts_a_valid_frame`, `tests/test_preprocessing.py::test_validate_rejects_missing_required_column`, `tests/test_preprocessing.py::test_validate_rejects_duplicated_column`, `tests/test_preprocessing.py::test_validate_rejects_unexpected_column`, `tests/test_full_data.py::test_real_dataset_matches_the_published_description`, `tests/test_integration.py::test_load_data_reports_every_file_problem_explicitly`
- **T02** : `tests/test_preprocessing.py::test_validate_rejects_string_typed_numeric_column`, `tests/test_preprocessing.py::test_validate_rejects_object_dtype_even_if_values_look_numeric`, `tests/test_preprocessing.py::test_validate_rejects_boolean_feature`, `tests/test_preprocessing.py::test_validate_rejects_non_dataframe_inputs`, `tests/test_preprocessing.py::test_validate_rejects_non_numeric_target`
- **T03** : `tests/test_preprocessing.py::test_standard_scaler`, `tests/test_preprocessing.py::test_preprocessor_output_is_finite_with_expected_dimensions`, `tests/test_preprocessing.py::test_preprocessor_imputes_missing_values_so_output_stays_finite`, `tests/test_preprocessing.py::test_preprocessor_keeps_dimensions_when_a_training_column_is_entirely_missing`, `tests/test_preprocessing.py::test_property_valid_inputs_always_give_finite_preprocessed_features`, `tests/test_preprocessing.py::test_amount_transform_clips_negative_values_before_the_logarithm`
- **T04** : `tests/test_prediction.py::test_prediction_probabilities_are_valid`, `tests/test_prediction.py::test_predict_fraud_output_contract`, `tests/test_prediction.py::test_predict_fraud_detects_model_contract_violations`, `tests/test_prediction.py::test_model_class_order_is_0_then_1`, `tests/test_robustness.py::test_property_any_in_contract_input_gives_a_valid_probability`
- **T05** : `tests/test_integration.py::test_split_is_disjoint_complete_and_stratified`, `tests/test_integration.py::test_end_to_end_run_uses_each_partition_only_for_its_role`, `tests/test_integration.py::test_pipeline_fit_never_sees_validation_or_test_rows`, `tests/test_integration.py::test_tuning_uses_validation_rows_only_and_picks_a_grid_value`
- **T06** : `tests/test_regression.py::test_two_trainings_with_same_seed_give_identical_probabilities`, `tests/test_regression.py::test_current_code_matches_the_frozen_ci_baseline`, `tests/test_integration.py::test_training_is_reproducible_across_independent_processes`, `tests/test_integration.py::test_split_is_reproducible_and_depends_on_the_seed`, `tests/test_integration.py::test_tune_model_is_deterministic`
- **T07** : `tests/test_ci_config.py::test_workflow_runs_on_pull_requests_and_pushes_to_main`, `tests/test_ci_config.py::test_workflow_runs_pytest_with_coverage_gate_on_two_python_versions`, `tests/test_ci_config.py::test_workflows_never_mask_failures`, `tests/test_ci_config.py::test_workflow_uploads_evidence_even_when_tests_fail_and_signals_failure`
- **T08** : `tests/test_preprocessing.py::test_validate_accepts_few_missing_values_with_warning`, `tests/test_preprocessing.py::test_validate_rejects_too_many_missing_values`, `tests/test_preprocessing.py::test_validate_rejects_any_missing_value_when_channel_forbids_them`, `tests/test_preprocessing.py::test_validate_rejects_infinite_values`, `tests/test_preprocessing.py::test_validate_rejects_target_values_other_than_0_and_1`, `tests/test_preprocessing.py::test_validate_single_observation_is_valid_and_keeps_dimensions`, `tests/test_preprocessing.py::test_validate_amount_boundaries`, `tests/test_preprocessing.py::test_validate_rejects_empty_dataframe`, `tests/test_preprocessing.py::test_missing_fraction_limit_is_inclusive_at_exactly_five_percent`
- **T09** : `tests/test_preprocessing.py::test_scaler_is_fitted_on_training_data_only`, `tests/test_integration.py::test_pipeline_fit_never_sees_validation_or_test_rows`, `tests/test_integration.py::test_split_is_disjoint_complete_and_stratified`, `tests/test_integration.py::test_deduplicate_removes_exact_duplicates_so_none_cross_partitions`, `tests/test_integration.py::test_end_to_end_run_uses_each_partition_only_for_its_role`
- **T10** : `tests/test_evaluation.py::test_select_threshold_respects_the_fpr_constraint_and_maximises_recall`, `tests/test_evaluation.py::test_select_threshold_is_monotonic_in_the_fpr_budget`, `tests/test_evaluation.py::test_select_threshold_rejects_invalid_inputs`, `tests/test_evaluation.py::test_threshold_table_is_monotonic`, `tests/test_prediction.py::test_threshold_boundary_is_inclusive`
- **T11** : `tests/test_evaluation.py::test_acceptance_criteria_are_well_formed`, `tests/test_evaluation.py::test_verdict_rules_critical_major_and_pass`, `tests/test_evaluation.py::test_criteria_flag_statistical_uncertainty`, `tests/test_integration.py::test_both_models_train_and_evaluate`, `tests/test_evaluation.py::test_meets_performance_criteria_checks_each_requirement`, `tests/test_evaluation.py::test_select_candidate_prefers_a_qualified_model_over_a_higher_pr_auc`, `tests/test_evaluation.py::test_select_candidate_takes_the_best_pr_auc_among_qualified_models`, `tests/test_evaluation.py::test_select_candidate_returns_no_pick_when_nobody_qualifies`
- **T12** : `tests/test_integration.py::test_saved_and_reloaded_pipeline_predicts_identically`, `tests/test_integration.py::test_reload_detects_a_tampered_artifact_before_deserialising`, `tests/test_integration.py::test_reload_refuses_an_artifact_without_hash_file`, `tests/test_integration.py::test_reload_missing_or_malformed_artifact_is_reported`
- **T13** : `tests/test_integration.py::test_dimensions_are_consistent_from_input_to_prediction`, `tests/test_preprocessing.py::test_preprocess_data_shapes_order_dtype_and_index`, `tests/test_prediction.py::test_predict_fraud_does_not_depend_on_column_order`, `tests/test_prediction.py::test_predict_fraud_single_row`
- **T14** : `tests/test_robustness.py::test_logistic_regression_sensitivity_respects_its_theoretical_bound`, `tests/test_robustness.py::test_sensitivity_grows_with_the_noise_amplitude`, `tests/test_robustness.py::test_small_perturbation_flips_few_decisions`, `tests/test_robustness.py::test_zero_perturbation_changes_nothing`, `tests/test_robustness.py::test_perturbation_protocol_matches_its_documentation`, `tests/test_robustness.py::test_decision_tree_predictions_are_row_independent_and_time_invariant`
- **T15** : `tests/test_robustness.py::test_extreme_amount_is_processed_with_an_alert`, `tests/test_robustness.py::test_absurd_amount_is_rejected`, `tests/test_robustness.py::test_missing_values_are_imputed_and_flagged`, `tests/test_robustness.py::test_covariate_shift_degrades_or_moves_scores`, `tests/test_monitoring.py::test_drift_report_levels`, `tests/test_monitoring.py::test_window_report_raises_alerts_on_drifted_inputs`
- **T16** : `tests/test_regression.py::test_current_code_matches_the_frozen_ci_baseline`, `tests/test_regression.py::test_a_degraded_model_is_rejected_by_the_gate`, `tests/test_regression.py::test_neutral_refactoring_is_accepted_by_the_gate`, `tests/test_regression.py::test_missing_baseline_is_an_error_and_is_never_recreated`, `tests/test_regression.py::test_write_baseline_refuses_to_overwrite_an_existing_reference`, `tests/test_regression.py::test_small_degradation_requires_investigation`, `tests/test_regression.py::test_large_degradation_is_rejected`, `tests/test_regression.py::test_ci_baseline_records_everything_required_by_the_assignment`
- **T17** : `tests/test_performance.py::test_single_prediction_latency_has_the_expected_order_of_magnitude`, `tests/test_performance.py::test_request_handler_latency_has_the_expected_order_of_magnitude`, `tests/test_performance.py::test_batch_throughput_has_the_expected_order_of_magnitude`, `tests/test_performance.py::test_serialized_model_is_small`
- **T18** : `tests/test_prediction.py::test_handler_reports_missing_fields`, `tests/test_prediction.py::test_handler_rejects_unexpected_fields_without_echoing_their_names`, `tests/test_prediction.py::test_handler_rejects_wrong_types`, `tests/test_prediction.py::test_handler_rejects_non_finite_json_literals`, `tests/test_prediction.py::test_handler_rejects_float_overflow_to_infinity`, `tests/test_prediction.py::test_handler_rejects_values_outside_the_data_contract`, `tests/test_prediction.py::test_handler_rejects_malformed_or_non_object_payloads`, `tests/test_prediction.py::test_handler_rejects_oversized_payload`, `tests/test_prediction.py::test_handler_masks_internal_errors`
- **T19** : `tests/test_security.py::test_no_secret_is_committed_in_the_repository`, `tests/test_security.py::test_secret_scanner_is_not_vacuous`, `tests/test_security.py::test_gitignore_protects_data_models_and_credentials`, `tests/test_security.py::test_validation_warnings_log_counts_not_values`, `tests/test_security.py::test_kaggle_csv_is_not_tracked_by_git`, `tests/test_security.py::test_no_large_file_is_versionable`, `tests/test_prediction.py::test_handler_never_logs_input_values`
- **T20** : `tests/test_ci_config.py::test_workflow_runs_ruff_lint_and_format_check`, `tests/test_ci_config.py::test_workflow_runs_pytest_with_coverage_gate_on_two_python_versions`, `tests/test_ci_config.py::test_workflow_uses_least_privilege_permissions`, `tests/test_ci_config.py::test_heavy_workflow_is_manual_or_scheduled_only`
- **T21** : `tests/test_evaluation.py::test_evaluate_model_matches_hand_computation`, `tests/test_evaluation.py::test_accuracy_is_misleading_on_imbalanced_data`, `tests/test_evaluation.py::test_undefined_metrics_are_nan_not_zero`, `tests/test_evaluation.py::test_precision_is_zero_without_alerts`, `tests/test_evaluation.py::test_evaluate_model_rejects_invalid_inputs`, `tests/test_evaluation.py::test_ranking_metrics_match_scikit_learn`, `tests/test_evaluation.py::test_average_precision_of_random_scores_is_close_to_prevalence`, `tests/test_evaluation.py::test_wilson_interval_known_values`
- **T22** : `tests/test_monitoring.py::test_psi_detects_mean_shift_and_scale_change`, `tests/test_monitoring.py::test_psi_edge_cases`, `tests/test_monitoring.py::test_window_report_is_quiet_on_a_stable_window`, `tests/test_monitoring.py::test_window_report_raises_alerts_on_drifted_inputs`, `tests/test_monitoring.py::test_alert_rate_validations`, `tests/test_monitoring.py::test_window_report_flags_a_collapse_of_the_alert_rate`, `tests/test_monitoring.py::test_alert_rate_rule_needs_both_a_significant_and_a_large_change`
- **T23** : `tests/test_evaluation.py::test_segment_report_matches_per_group_metrics_and_flags_small_samples`, `tests/test_evaluation.py::test_segment_report_rejects_misaligned_groups`, `tests/test_evaluation.py::test_dataset_contract_contains_no_demographic_attribute`

## 5. Critères d'acceptation et de rejet

Définis à partir d'exigences métier et **figés avant toute exécution sur le jeu de test** (`app/config.py`).
Règle de décision : un critère **critique** en échec (ou non mesuré) ⇒ REJETÉ ; un critère **majeur** en échec ⇒ À INVESTIGUER ;
sinon APTE À UN PILOTE (sous conditions), les échecs mineurs devenant des réserves.

| ID | Critère | Seuil | Gravité | Justification |
|---|---|---|---|---|
| CA01 | Rappel au seuil figé (jeu de test) | >= 0.75 | critique | Au moins 3 fraudes sur 4 interceptées ; en deçà, la perte résiduelle rend le modèle peu utile face à des règles expertes. |
| CA02 | Taux de faux positifs (jeu de test) | <= 0.001 | critique | Au plus 1 transaction légitime sur 1 000 bloquée à tort (≈ 140 fausses alertes/jour pour 142 500 transactions/jour). |
| CA03 | Précision au seuil figé (jeu de test) | >= 0.5 | majeur | Au moins une alerte sur deux doit être une vraie fraude pour que les analystes gardent confiance dans les alertes. |
| CA04 | PR-AUC / Average Precision (jeu de test) | >= 0.65 | majeur | Qualité de classement indépendante du seuil ; ≈ 380 fois la prévalence (0,17 %), seuil d'un modèle réellement utile. |
| CA05 | ROC-AUC (jeu de test) | >= 0.95 | majeur | Pouvoir de séparation global minimal attendu d'un détecteur de fraude sur ce type de données. |
| CA06 | Écart de généralisation PR-AUC (entraînement − test) | <= 0.15 | majeur | Un écart supérieur à 0,15 signale un sur-apprentissage. |
| CA07 | Reproductibilité : écart max entre deux entraînements | <= 1e-09 | critique | Mêmes données + mêmes graines = mêmes probabilités (tolérance numérique). |
| CA08 | Cohérence avant/après sauvegarde-rechargement | <= 1e-12 | critique | L'artefact déployé doit se comporter exactement comme celui qui a été testé. |
| CA09 | Robustesse : taux de changement de classe (perturbation modérée) | <= 0.005 | majeur | Moins de 0,5 % des décisions peuvent changer sous un bruit réaliste (0,05 σ sur V1-V28 ; ±5 % sur Amount). |
| CA10 | Robustesse : baisse de rappel (perturbation modérée) | <= 0.05 | majeur | Le bruit ne doit pas faire perdre plus de 5 points de rappel. |
| CA11 | Entrées invalides acceptées à tort | <= 0 | critique | Toute entrée hors contrat doit être rejetée explicitement. |
| CA12 | Latence P95, prédiction isolée (fonction de service, ms) | <= 20.0 | majeur | Budget alloué au scoring dans une autorisation de paiement (réponse globale visée < 100-200 ms). |
| CA13 | Latence P95, gestionnaire de requête (JSON + validation + score, ms) | <= 50.0 | majeur | Marge pour la couche HTTP et le réseau dans le budget global. |
| CA14 | Débit en lot de 1 000 transactions (tx/s) | >= 1000.0 | mineur | ≥ 10 fois le pic estimé (~100 tx/s) pour le traitement par lots. |
| CA15 | Taille du modèle sérialisé (Mo) | <= 10.0 | mineur | Chargement rapide et empreinte réduite en conteneur. |
| CA16 | Surcoût mémoire pendant l'inférence en lot (Mo) | <= 500.0 | mineur | Compatible avec une limite de conteneur de 1 Go. |
| CA17 | Secrets détectés dans le dépôt | <= 0 | critique | Aucun identifiant, clé ou jeton versionné. |
| CA18 | Valeurs sensibles retrouvées dans les journaux | <= 0 | critique | Les journaux ne contiennent ni valeurs d'entrée ni données de transaction. |
| CA19 | Tests automatisés en échec | <= 0 | critique | Aucune régression connue tolérée. |
| CA20 | Couverture de code (%) | >= 85.0 | majeur | Seuil usuel d'une suite de tests crédible sur le code critique. |
| CA21 | Erreurs de style/qualité Ruff | <= 0 | majeur | Code homogène et analysé statiquement. |
| CA22 | Barrière de non-régression : scénarios de référence mal classés | <= 0 | majeur | Le contrôle doit accepter un refactoring neutre et rejeter une dégradation. |

## 6. Outils et preuves à conserver

| Outil | Usage | Preuve |
|---|---|---|
| Pytest, pytest-cov | exécution, couverture lignes + branches | `reports/junit.xml`, `reports/coverage.json`, `reports/ci_evidence/pytest_success.txt` |
| Hypothesis | tests de propriétés (mode déterministe) | tests `test_property_*` |
| Ruff | analyse statique et format | `reports/ci_evidence/ruff_check.txt` |
| scikit-learn, joblib | pipeline, sérialisation, SHA-256 | `models/*.json`, `reports/results.json` |
| `scripts/run_mutation.py` | les tests détectent-ils de vrais défauts ? | `reports/mutation.json`, `reports/ci_evidence/pytest_intentional_failure.txt` |
| GitHub Actions | exécution automatique, artefacts | `.github/workflows/tests.yml`, `experiments.yml` |

## 7. Hypothèses de cadrage à valider avec le métier

Volume ≈ 142 400 transactions/jour ; coût d'une fausse alerte 5 € ; plafond de friction 1 fausse alerte pour
1 000 transactions normales ; budget de scoring 20 ms dans une autorisation de paiement.

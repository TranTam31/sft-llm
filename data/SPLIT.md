# Báo cáo tách seen/unseen

Sinh tự động bởi `python -m src.data.split`. Đừng sửa tay; chạy lại để cập nhật.

Embedder `BAAI/bge-small-en-v1.5` @ `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 50 cụm (spherical k-means, seed 0). Tool gần như trùng nhau (cosine ≥ 0.95) được dồn về cùng một cụm: 494 tool đổi cụm, khối lớn nhất 737 tool.

Chọn 8 cụm unseen qua 3000 lần thử: chỉ nhận phương án đạt mọi ràng buộc cứng, xếp theo khoảng cách phân phối cộng rò rỉ ranh giới (cặp tool hai phía có cosine ≥ 0.85, trọng số 1.0). Phương án được dùng đứng thứ 1 trong bảng xếp hạng (465 phương án đạt ràng buộc lúc tìm); 0 phương án trước bị loại khi kiểm lại trên pool cuối.

Sau khi chọn cụm, tool seen có cosine ≥ 0.92 với một tool unseen, hoặc có tên gần giống (trigram ký tự ≥ 0.6 sau khi bỏ dấu gạch và chữ hoa) và cosine ≥ 0.85, bị kéo sang unseen (lặp tới khi hết): 75 tool qua 3 vòng.

Ứng viên unseen là các khối cụm: cụm có tâm gần nhau (cosine ≥ 0.92) gộp thành một khối và chỉ được chọn nguyên khối. Khối có độ đồng nhất (cosine trung bình giữa các tool) < 0.6 hoặc bị loại tay trong `split.excluded_clusters` không được chọn.

Cụm ghim unseen sau khi duyệt tay (`split.pinned_unseen`): 4, 7, 14, 23, 35, 46; thuật toán chỉ chọn thêm phần còn lại.

## Kiểm tra trên pool cuối

| Điều kiện | Giá trị | Ngưỡng | Kết quả |
|---|---|---|---|
| Mẫu unseen nhóm single | 1628 | ≥ 250 | ĐẠT |
| Mẫu unseen nhóm select | 1086 | ≥ 250 | ĐẠT |
| Mẫu unseen nhóm parallel | 2697 | ≥ 250 | ĐẠT |
| Mẫu unseen nhóm multi_turn | 5852 | ≥ 250 | ĐẠT |
| Mẫu unseen nhóm clarify_candidate | 1218 | ≥ 250 | ĐẠT |
| Tỉ lệ mẫu tool mất khỏi train (unseen + lẫn) | 23.5% | ≤ 30% | ĐẠT |
| Khoảng cách phân phối seen/unseen (nhóm, số tool, số tham số) | 0.678 | chỉ báo | – |

**Kết luận: đạt mọi ràng buộc cứng.**

## Số mẫu theo pool

| Nhóm | train | seen_eval | unseen_eval | chat_eval | Bỏ vì lẫn seen/unseen |
|---|---|---|---|---|---|
| call_no_response | 359 | 0 | 0 | 0 | 12 |
| chat | 15885 | 0 | 0 | 1000 | 0 |
| chat_no_tools | 21713 | 0 | 0 | 0 | 0 |
| clarify_candidate | 5267 | 600 | 1218 | 0 | 778 |
| multi_turn | 20408 | 600 | 5852 | 0 | 2797 |
| no_call | 171 | 600 | 2118 | 0 | 0 |
| parallel | 20361 | 600 | 2697 | 0 | 6275 |
| select | 9717 | 600 | 1086 | 0 | 7119 |
| single | 5724 | 600 | 1628 | 0 | 0 |

## Miền unseen cần duyệt

Mỗi cụm unseen là một miền chức năng bị giữ khỏi train. Duyệt: tool trong miền có cùng chức năng không, và các tool seen gần ranh giới có thực ra cùng chức năng (cần gộp hoặc loại) không.

### Cụm 29 (chọn thêm): 282 tool, được cung cấp 23492 lần

Tool hay gặp nhất: `sort_numbers`, `calculate_standard_deviation`, `is_perfect_square`, `std_deviation`, `find_n_largest_numbers`, `find_pairs_with_sum`, `bacterial_growth`, `calculate_median`, `calculate_factorial`, `find_max_subarray_sum`, `chi_square_independence_test`, `merge_dictionaries`.

Cụm seen giáp ranh (số cặp tool gần nhau): cụm 2 (411 cặp: `generate_password`, `generate_random_number`, `generate_random_password`); cụm 10 (9 cặp: `get_definition`, `find_longest_word`, `reverse_words`); cụm 32 (7 cặp: `calculate_bmi`, `calculate_area`, `cell_density`).

| Cosine | Tool unseen | Tool seen gần nhất | Cụm seen |
|---|---|---|---|
| 0.879 | `count_characters` | `count_words` | 10 |
| 0.863 | `count_characters` | `count_words` | 10 |
| 0.857 | `count_characters` | `count_words` | 10 |
| 0.857 | `calculate_square` | `calculate_area` | 32 |
| 0.854 | `calculate_sum` | `calculate_total` | 32 |

### Cụm 36 (chọn thêm): 137 tool, được cung cấp 5830 lần

Tool hay gặp nhất: `is_leap_year`, `is_rotation`, `is_palindrome`, `is_valid_palindrome`, `is_valid_ip_address`, `is_valid_parentheses`, `is_anagram`, `find_first_non_repeating_char`, `is_valid_sudoku`, `is_anagram_phrase`, `find_longest_palindromic_substring`, `is_subset`.

Cụm seen giáp ranh (số cặp tool gần nhau): cụm 10 (56 cặp: `get_definition`, `find_longest_word`, `reverse_words`); cụm 2 (4 cặp: `generate_password`, `generate_random_number`, `generate_random_password`); cụm 5 (2 cặp: `loginuser`, `getuserbyname`, `search`).

| Cosine | Tool unseen | Tool seen gần nhất | Cụm seen |
|---|---|---|---|
| 0.911 | `check_spelling` | `check_word_meaning` | 10 |
| 0.910 | `check_word_spell` | `check_word_meaning` | 10 |
| 0.909 | `check_word_spelling` | `check_word_meaning` | 10 |
| 0.906 | `check_spelling` | `check_word_meaning` | 10 |
| 0.899 | `check_spelling` | `check_word_definition` | 10 |

### Cụm 46 (ghim): 321 tool, được cung cấp 4418 lần

Tool hay gặp nhất: `calculate_tip`, `calculate_tip_split`, `calculate_tip_amount`, `calculate_tip_percentage`, `calculate_tip_share`, `calculate_bill`, `calculate_invoice_total`, `calculate_tip_percent`, `calculate_payment_due`, `calculate_tips`.

Cụm seen giáp ranh (số cặp tool gần nhau): cụm 32 (13 cặp: `calculate_bmi`, `calculate_area`, `cell_density`); cụm 43 (4 cặp: `generate_qr_code`, `generate_invoice`, `create_invoice`); cụm 3 (1 cặp: `calculate_discount`, `calculate_discounted_price`, `calculate_profit`).

| Cosine | Tool unseen | Tool seen gần nhất | Cụm seen |
|---|---|---|---|
| 0.914 | `calculate_invoice_total` | `create_invoice` | 43 |
| 0.895 | `calculate_bill` | `calculate_total` | 32 |
| 0.871 | `calculate_tip` | `calculate_percentage` | 32 |
| 0.862 | `calculate_tip` | `calculate_percentage` | 32 |
| 0.860 | `calculate_tip` | `calculate_percentage` | 32 |

### Cụm 14 (ghim): 187 tool, được cung cấp 3236 lần

Tool hay gặp nhất: `fixtures`, `tournaments`, `super_lig_standings`, `runner_up`, `primeira_liga_standings`, `tournament_list`, `seasonendconstructorstanding`, `get_all_players`, `eredivisie_standings`, `tournament_list_wta`, `teamlastevents`, `games_esports`.

Cụm seen giáp ranh (số cặp tool gần nhau): cụm 37 (9 cặp: `get_product`, `get_order`, `get_by_id`); cụm 16 (5 cặp: `get_pokemon_move_info`, `get_dna_sequence`, `getorderbyid`).

| Cosine | Tool unseen | Tool seen gần nhất | Cụm seen |
|---|---|---|---|
| 0.886 | `national_team_statistics` | `player_data` | 16 |
| 0.881 | `getteamroster` | `getteamstats` | 37 |
| 0.874 | `games_esports` | `eventgames` | 37 |
| 0.865 | `statistics_esports` | `event_data` | 16 |
| 0.857 | `competitions_standings` | `stagecompetitorstandings` | 37 |

### Cụm 35 (ghim): 119 tool, được cung cấp 2334 lần

Tool hay gặp nhất: `send_email`, `send_sms`, `message_send`, `send_notification`, `post_social_media_status`, `send_message`, `send_push_notification`, `post_to_social_media`, `post_tweet`, `create_email`, `post_social_media`, `save_contact`.

Cụm seen giáp ranh (số cặp tool gần nhau): cụm 21 (25 cặp: `create_user`, `generate_username`, `create_user_account`); cụm 9 (3 cặp: `reverse_string`, `create_histogram`, `generate_uuid`); cụm 13 (2 cặp: `create_calendar_event`, `create_todo`, `create_note`).

| Cosine | Tool unseen | Tool seen gần nhất | Cụm seen |
|---|---|---|---|
| 0.901 | `save_contact` | `create_contact` | 21 |
| 0.868 | `post_to_social_media` | `schedule_social_media_post` | 13 |
| 0.862 | `create_email` | `create_contact` | 21 |
| 0.859 | `post_social_media_status` | `schedule_social_media_post` | 13 |
| 0.851 | `create_email` | `create_contact` | 21 |

### Cụm 4 (ghim): 159 tool, được cung cấp 2267 lần

Tool hay gặp nhất: `matchschedules`, `fixtures_by_date`, `game_odds_by_category_schedule`, `basketball_match_statistics`, `get_sports_scores`, `daily_match_list_all`, `get_match`, `competitions`, `statistics`, `trainers_win_rate`, `daily_match_list_live`, `fixtures`.

Cụm seen giáp ranh (số cặp tool gần nhau): cụm 28 (5 cặp: `can_attend_all_meetings`, `predict_evolution_rate`, `get_current_time`); cụm 37 (4 cặp: `get_product`, `get_order`, `get_by_id`); cụm 16 (2 cặp: `get_pokemon_move_info`, `get_dna_sequence`, `getorderbyid`).

| Cosine | Tool unseen | Tool seen gần nhất | Cụm seen |
|---|---|---|---|
| 0.902 | `search` | `basketball_search` | 34 |
| 0.896 | `get_match` | `getpercentage` | 26 |
| 0.889 | `drawings_between_dates` | `drawing_by_date` | 28 |
| 0.889 | `match_scorecard` | `match_details_by_id` | 16 |
| 0.874 | `matchschedules` | `nhl_schedules` | 28 |

### Cụm 23 (ghim): 100 tool, được cung cấp 2196 lần

Tool hay gặp nhất: `is_valid_email`, `check_email_availability`, `validate_email`, `emailvalidation`, `check_email`, `email`, `check_for_disposable_emails`, `email_validate_regex`, `email_verifier`, `phone_number`, `mailcheck`, `ip_address`.

Cụm seen giáp ranh (số cặp tool gần nhau): cụm 30 (1 cặp: `get_holidays_by_year`, `get_countries`, `auto_complete`); cụm 49 (1 cặp: `search`, `whois`, `obd2_code`).

| Cosine | Tool unseen | Tool seen gần nhất | Cụm seen |
|---|---|---|---|
| 0.881 | `phone_number` | `validate` | 30 |

### Cụm 7 (ghim): 102 tool, được cung cấp 1612 lần

Tool hay gặp nhất: `translate_text`, `get_translation`, `translate`, `detect_language`, `translation`, `shakespeare`, `v1_textlanguage`, `transliterate_any_to_latin`, `replace`, `languagedetection`, `tts`, `get_price_info_for_word_count`.

Cụm seen giáp ranh (số cặp tool gần nhau): cụm 10 (38 cặp: `get_definition`, `find_longest_word`, `reverse_words`); cụm 1 (17 cặp: `analyze_sentiment`, `structural_analysis`, `analyze_social_group_dynamics`).

| Cosine | Tool unseen | Tool seen gần nhất | Cụm seen |
|---|---|---|---|
| 0.911 | `get_translation` | `get_definition` | 10 |
| 0.910 | `get_translation` | `get_definition_by_language` | 10 |
| 0.900 | `get_translation` | `get_definition` | 10 |
| 0.896 | `get_translation` | `get_definition` | 10 |
| 0.890 | `identify_language` | `analyze_text_language` | 1 |

Tool bị kéo từ cụm seen sang unseen (hay gặp nhất): `generate_random_number`, `match`, `custom_dice`, `regular_dice`, `matchh2hduel`, `search`, `fixtures_by_date`, `custom_die`, `get_random_number`, `teamimage`, `generate_random_numbers`, `teams_list`, `drawings_between_dates`, `playerstatisticsseasons`, `game`.

## Khối gộp và khối bị loại khỏi ứng viên

| Khối (cụm) | Độ đồng nhất | Bị loại vì | Tool hay gặp nhất |
|---|---|---|---|
| 1 | 0.722 | gần tạp: gom theo động từ analyze (sentiment, ảnh, t-test, SEO) | `analyze_sentiment`, `structural_analysis` |
| 9 | 0.575 | độ đồng nhất 0.575 < 0.6 | `reverse_string`, `create_histogram` |
| 16, 26, 37, 42, 47, 49 | 0.545 | độ đồng nhất 0.545 < 0.6 | `get_pokemon_move_info`, `get_dna_sequence`, `title_v2_get_business`, `whois`, `get_product`, `get_order`, `getlearningresources`, `getnews` |
| 22 | 0.607 | miền mạng xã hội trải sang cụm seen 5 và 37 (tweet, post) | `video_comments`, `search` |
| 30 | 0.631 | gần tạp: quốc gia, ngày lễ lẫn danh mục, OTT, tab | `get_holidays_by_year`, `get_countries` |
| 33 | 0.601 | IP/địa lý giáp thời tiết (cụm 28) và khối tạp | `get_ip_zipcode`, `predict_forest_area` |
| 34 | 0.752 | tạp: nhiều chức năng tìm kiếm khác nhau dính nhau qua chữ search | `search`, `search_jobs` |
| 38 | 0.633 | miền tài chính tách đôi với cụm 12 (giá, lịch sử chứng khoán) | `latest_rates`, `time_series` |
| 4, 14 | 0.636 | – | `matchschedules`, `fixtures_by_date`, `fixtures`, `tournaments` |

## Khử trùng lặp train với eval

MinHash 128 hoán vị, shingle 3 từ, ngưỡng Jaccard 0.7; so trên các lượt user.

| Nhóm | Trùng exact | Gần trùng |
|---|---|---|
| chat | 15 | 100 |
| chat_no_tools | 0 | 2 |
| clarify_candidate | 4976 | 2644 |
| multi_turn | 11732 | 5735 |
| no_call | 11966 | 125 |
| parallel | 3 | 18 |
| select | 34 | 71 |
| single | 25 | 46 |

## Cụm chức năng

| Cụm | Unseen | Độ đồng nhất | Số tool | Tool bị kéo sang unseen | Số lần được cung cấp | Tool hay gặp nhất |
|---|---|---|---|---|---|---|
| 29 | ✔ | 0.629 | 282 |  | 23492 | `sort_numbers`, `calculate_standard_deviation`, `is_perfect_square`, `std_deviation`, `find_n_largest_numbers`, `find_pairs_with_sum`, `bacterial_growth`, `calculate_median` |
| 32 |  | 0.711 | 503 | 1 | 15974 | `calculate_bmi`, `calculate_area`, `cell_density`, `final_velocity`, `triangle_area`, `calculate_shipping_cost`, `calculate_gpa`, `batting_average` |
| 47 |  | 0.557 | 353 | 2 | 11058 | `simulate_query_database`, `binary_search`, `flatten_list`, `split_list`, `project_population`, `remove_duplicates`, `place_safeway_order`, `get_products_in_category` |
| 39 |  | 0.795 | 752 |  | 9854 | `calculate_loan_payment`, `calculate_mortgage_payment`, `calculate_tax`, `calculate_investment_return`, `future_investment_value`, `calculate_interest`, `project_investment_growth`, `california_alimony` |
| 37 |  | 0.711 | 357 | 5 | 9806 | `get_product`, `get_order`, `get_by_id`, `matchdetails`, `albums_get_related_artist`, `prematchform`, `categorytournaments`, `artist_albums` |
| 49 |  | 0.585 | 352 | 3 | 9009 | `search`, `whois`, `obd2_code`, `science`, `title_get_technical`, `get_all_kfc_locations_by_state`, `search_for_any_category_or_star`, `airports_direct_routes_for_an_airport` |
| 26 |  | 0.537 | 374 |  | 8240 | `title_v2_get_business`, `whois`, `current_block_number`, `get_artist_s_albums`, `software_assets`, `account_transaction_history`, `v1_whois`, `search` |
| 2 |  | 0.857 | 411 | 52 | 8031 | `generate_password`, `generate_random_number`, `generate_random_password`, `generate_random_string`, `getrandompassword`, `get_random_number`, `generate_random_numbers`, `generate_password_strength` |
| 17 |  | 0.732 | 375 |  | 7887 | `calculate_distance`, `euclidean_distance`, `calculate_electric_field`, `cosine_similarity`, `note_duration`, `light_travel_time`, `measure_distance`, `calculate_fuel_efficiency` |
| 38 |  | 0.633 | 309 |  | 7320 | `latest_rates`, `time_series`, `stockprice`, `historical_rates`, `spac_price_targets`, `ticker`, `yh_historical`, `market_capitalization` |
| 33 |  | 0.601 | 230 |  | 6643 | `get_ip_zipcode`, `predict_forest_area`, `get_ip_location`, `geocode_city`, `get_city_from_zipcode`, `ip_lookup`, `geocode`, `by_zip_code` |
| 0 |  | 0.783 | 240 |  | 6362 | `convert_currency`, `convert_temperature`, `format_date`, `convert_cooking_measurements`, `get_exchange_rate`, `v1_exchangerate`, `calculate_fahrenheit_to_celsius`, `exchange` |
| 28 |  | 0.615 | 348 | 2 | 6093 | `can_attend_all_meetings`, `predict_evolution_rate`, `get_current_time`, `drawing_by_date`, `time_zone_api`, `billboard_hot_100`, `independent_albums`, `artist_100` |
| 36 | ✔ | 0.657 | 137 |  | 5830 | `is_leap_year`, `is_rotation`, `is_palindrome`, `is_valid_palindrome`, `is_valid_ip_address`, `is_valid_parentheses`, `is_anagram`, `find_first_non_repeating_char` |
| 16 |  | 0.610 | 237 | 2 | 5509 | `get_pokemon_move_info`, `get_dna_sequence`, `getorderbyid`, `getpetbyid`, `title_details`, `theme_detail`, `stores_get_details`, `mlem_id` |
| 43 |  | 0.745 | 850 |  | 5376 | `generate_qr_code`, `generate_invoice`, `create_invoice`, `generate_barcode`, `qr_code`, `generate_qrcode`, `v1_qrcode`, `qrcode` |
| 13 |  | 0.768 | 826 |  | 5314 | `create_calendar_event`, `create_todo`, `create_note`, `create_event`, `schedule_meeting`, `create_task`, `create_reminder`, `create_todo_list` |
| 11 |  | 0.822 | 833 |  | 5162 | `get_movie_details`, `search_movies`, `search_movie`, `get_movie_recommendations`, `search_by_genre`, `find_movie`, `search_by_name`, `get_movies_by_director` |
| 31 |  | 0.899 | 179 |  | 5061 | `calculate_age`, `calculate_age_difference`, `last_menstrual_period_lmp`, `age_calculator`, `calculate_age_in_days`, `calculate_age_in_months`, `calculate_lifespan`, `calculate_age_in_hours` |
| 5 |  | 0.639 | 225 |  | 4993 | `loginuser`, `getuserbyname`, `search`, `get_user_profile`, `get_user_playlists`, `get_user_info`, `get_sample_userprofile`, `search_user` |
| 3 |  | 0.876 | 431 |  | 4774 | `calculate_discount`, `calculate_discounted_price`, `calculate_profit`, `calculate_profit_margin`, `calculate_sales_tax`, `calculate_tax`, `calculate_total_price`, `calculate_discounted_percentage` |
| 46 | ✔ | 0.922 | 321 |  | 4418 | `calculate_tip`, `calculate_tip_split`, `calculate_tip_amount`, `calculate_tip_percentage`, `calculate_tip_share`, `calculate_bill`, `calculate_invoice_total`, `calculate_tip_percent` |
| 48 |  | 0.782 | 473 |  | 4246 | `get_news`, `get_news_headlines`, `search_news`, `top_headlines`, `get_news_articles`, `search`, `get_articles`, `health` |
| 30 |  | 0.631 | 170 |  | 4159 | `get_holidays_by_year`, `get_countries`, `auto_complete`, `get_regions`, `origin`, `countries_detail`, `getallcountry`, `ott_providers` |
| 24 |  | 0.693 | 271 | 1 | 4071 | `get_random_joke`, `generate_random_quote`, `get_random_fact`, `generate_random_color`, `get_random_quote`, `generate_random_fact`, `generate_random_name`, `random` |
| 22 |  | 0.607 | 183 |  | 3879 | `video_comments`, `search`, `download_stream`, `search_suggestions`, `get_channel_points_context`, `video_details`, `channel_playlists`, `video` |
| 42 |  | 0.629 | 157 |  | 3747 | `getlearningresources`, `getnews`, `getpeople`, `getpastevents`, `getfutureevents`, `top_players_by_matchmaking`, `getcommunities`, `get_all_posts` |
| 19 |  | 0.874 | 562 |  | 3678 | `search_recipes`, `search_recipe`, `get_recipe`, `search_keto_recipe`, `keto_recipes_by_difficulty`, `find_recipe`, `recipe_search_and_recommendations`, `search_for_recipes` |
| 9 |  | 0.575 | 223 | 1 | 3657 | `reverse_string`, `create_histogram`, `generate_uuid`, `encrypt_text`, `convert_hsl`, `generate`, `hsl_to_rgb`, `go` |
| 10 |  | 0.709 | 122 | 3 | 3422 | `get_definition`, `find_longest_word`, `reverse_words`, `word_frequency`, `count_vowels`, `check_word_count`, `get_synonyms`, `count_words` |
| 14 | ✔ | 0.664 | 187 |  | 3236 | `fixtures`, `tournaments`, `super_lig_standings`, `runner_up`, `primeira_liga_standings`, `tournament_list`, `seasonendconstructorstanding`, `get_all_players` |
| 12 |  | 0.855 | 189 |  | 3200 | `get_stock_price`, `stock_get_stock_price`, `stock_company_overview`, `check_stock_price`, `full_stock_price`, `get_stock_prices`, `analyze_stock_performance`, `analyze_stock_market` |
| 1 |  | 0.722 | 275 | 1 | 3185 | `analyze_sentiment`, `structural_analysis`, `analyze_social_group_dynamics`, `independent_samples_t_test`, `analyze_image`, `onpage_seo_analyze`, `analyze_website`, `text_sentiment_analysis` |
| 18 |  | 0.890 | 519 |  | 3100 | `search_books`, `search_book`, `search_title`, `search_books_by_name`, `search`, `search_author`, `get_book_details`, `find_book` |
| 34 |  | 0.752 | 446 | 1 | 2860 | `search`, `search_jobs`, `search_images`, `image_search`, `web_search`, `searchjobs`, `search_searchterm`, `search_autocomplete_api` |
| 35 | ✔ | 0.809 | 119 |  | 2334 | `send_email`, `send_sms`, `message_send`, `send_notification`, `post_social_media_status`, `send_message`, `send_push_notification`, `post_to_social_media` |
| 4 | ✔ | 0.646 | 159 |  | 2267 | `matchschedules`, `fixtures_by_date`, `game_odds_by_category_schedule`, `basketball_match_statistics`, `get_sports_scores`, `daily_match_list_all`, `get_match`, `competitions` |
| 23 | ✔ | 0.763 | 100 |  | 2196 | `is_valid_email`, `check_email_availability`, `validate_email`, `emailvalidation`, `check_email`, `email`, `check_for_disposable_emails`, `email_validate_regex` |
| 25 |  | 0.762 | 505 |  | 2108 | `play_music`, `get_lyrics`, `get_song_lyrics`, `search_music`, `get_radio_playlist`, `search_song`, `create_playlist`, `all_songs_from_artist` |
| 20 |  | 0.878 | 347 |  | 1735 | `search_restaurants`, `whole_foods_order`, `find_nearby_restaurants`, `find_restaurants`, `search_restaurant`, `find_nearest_restaurant`, `search_restaurants_by_local_authority`, `restaurants_v2_auto_complete` |
| 21 |  | 0.726 | 314 |  | 1717 | `create_user`, `generate_username`, `create_user_account`, `create_contact`, `create_account`, `generate_random_username`, `create_user_profile`, `create_poll` |
| 7 | ✔ | 0.834 | 102 |  | 1612 | `translate_text`, `get_translation`, `translate`, `detect_language`, `translation`, `shakespeare`, `v1_textlanguage`, `transliterate_any_to_latin` |
| 45 |  | 0.853 | 192 |  | 1529 | `min_meeting_rooms`, `reserve_hotel_room`, `is_hotel_available`, `search_hotels`, `find_hotels`, `book_hotel`, `list_of_room_facility_types`, `list_of_hotel_types` |
| 15 |  | 0.784 | 226 |  | 1474 | `search_product`, `search`, `tmall_search_by_keyword`, `products_detail`, `product_categorization`, `searchinventory`, `products_search`, `products_search_by_barcode` |
| 27 |  | 0.711 | 186 |  | 840 | `find_nearest_gas_station`, `find_nearby_places`, `get_stations`, `nearest_metro_station`, `nearby_superchargers`, `get_stations_within_1_km`, `superfund_search`, `ocean_conditions_closest_to_latitude_longitude` |
| 40 |  | 0.745 | 165 |  | 793 | `calculate_calorie_intake`, `track_package`, `track_calories`, `track_expenses`, `calculate_calories_burned`, `target_heart_rate`, `activities`, `calculate_calories` |
| 8 |  | 0.716 | 149 |  | 736 | `check_flight_status`, `directions_between_2_locations`, `find_shortest_path`, `book_flight`, `get_directions`, `get_trip_plan`, `airportstatistics`, `bus_trolley_routes` |
| 44 |  | 0.670 | 79 | 1 | 547 | `find_files`, `execute_program`, `upload_file`, `retrieve_file`, `execute_command`, `take_screenshot`, `execute_shell_command`, `get_file_size` |
| 6 |  | 0.771 | 61 |  | 483 | `go`, `v1_cocktail`, `feeds_auto_complete`, `explore`, `generate_recipe`, `detail`, `review`, `get_recipe_details` |
| 41 |  | 0.807 | 40 |  | 54 | `recommend_books`, `get_song_recommendations`, `generate_recommendations`, `rate_movie`, `get_movie_recommendations`, `get_music_recommendations`, `get_recommendations`, `get_book_recommendations` |

## Tool unseen gần tool seen nhất (cosine ≥ 0.85, 93 tool)

| Cosine | Tool unseen | Cụm | Tool seen gần nhất | Cụm |
|---|---|---|---|---|
| 0.914 | `calculate_invoice_total` | 46 | `create_invoice` | 43 |
| 0.911 | `match` | 37 | `matchdetails` | 37 |
| 0.911 | `check_spelling` | 36 | `check_word_meaning` | 10 |
| 0.911 | `get_translation` | 7 | `get_definition` | 10 |
| 0.910 | `count_occurrences` | 10 | `get_word_count` | 10 |
| 0.910 | `get_translation` | 7 | `get_definition_by_language` | 10 |
| 0.910 | `check_word_spell` | 36 | `check_word_meaning` | 10 |
| 0.909 | `check_word_spelling` | 36 | `check_word_meaning` | 10 |
| 0.906 | `check_spelling` | 36 | `check_word_meaning` | 10 |
| 0.902 | `search` | 4 | `basketball_search` | 34 |
| 0.901 | `save_contact` | 35 | `create_contact` | 21 |
| 0.900 | `get_translation` | 7 | `get_definition` | 10 |
| 0.899 | `check_spelling` | 36 | `check_word_definition` | 10 |
| 0.896 | `get_translation` | 7 | `get_definition` | 10 |
| 0.896 | `get_match` | 4 | `getpercentage` | 26 |
| 0.895 | `check_word_spelling` | 36 | `check_word_definition` | 10 |
| 0.895 | `calculate_bill` | 46 | `calculate_total` | 32 |
| 0.890 | `identify_language` | 7 | `analyze_text_language` | 1 |
| 0.889 | `drawings_between_dates` | 4 | `drawing_by_date` | 28 |
| 0.889 | `match_scorecard` | 4 | `match_details_by_id` | 16 |
| 0.889 | `match` | 37 | `matchvotes` | 37 |
| 0.888 | `teamimage` | 37 | `playerimage` | 37 |
| 0.888 | `detect_language` | 7 | `analyze_text_language` | 1 |
| 0.886 | `national_team_statistics` | 14 | `player_data` | 16 |
| 0.886 | `detect_language` | 7 | `analyze_text_language` | 1 |
| 0.884 | `check_word_spellings` | 36 | `check_word_definition` | 10 |
| 0.883 | `check_spelling` | 36 | `check_word_definition` | 10 |
| 0.881 | `getteamroster` | 14 | `getteamstats` | 37 |
| 0.881 | `phone_number` | 23 | `validate` | 30 |
| 0.880 | `playerstatisticsseasons` | 37 | `get_specific_player` | 37 |
| 0.879 | `count_characters` | 29 | `count_words` | 10 |
| 0.878 | `find_word_occurrences` | 10 | `get_word_count` | 10 |
| 0.874 | `games_esports` | 14 | `eventgames` | 37 |
| 0.874 | `matchschedules` | 4 | `nhl_schedules` | 28 |
| 0.873 | `generate_random_numbers` | 2 | `generate_lottery_numbers` | 2 |
| 0.871 | `stock_v2_get_competitors` | 4 | `stock_v2_get_trading_information` | 16 |
| 0.871 | `calculate_tip` | 46 | `calculate_percentage` | 32 |
| 0.869 | `check_word_availability` | 36 | `check_username_availability` | 5 |
| 0.868 | `check_password_strength` | 36 | `generate_password_strength` | 2 |
| 0.868 | `post_to_social_media` | 35 | `schedule_social_media_post` | 13 |

## Hash

| File | SHA-256 |
|---|---|
| `train.jsonl` | `085f4138c8fd69bb50d34fc12733f0760a93e1ec77df9001556a883910b9a315` |
| `seen_eval.jsonl` | `707198b47c4ea05a7df70035113b8eb6cc199896bc0befdf63ef0006855aab97` |
| `unseen_eval.jsonl` | `74c6a860d051c56d0186fac200355161ed067d896d51b64ffa5d58b2d3397a75` |
| `chat_eval.jsonl` | `78408bea072c41ace61fc1989f6e5fd3316ef0429c79e2b85c5230ba0d90acd4` |
| `tools.jsonl` | `9de4de26f79ada4d8008acd6ffa6b4a18b114943ecf5ac60010016c44ea81990` |

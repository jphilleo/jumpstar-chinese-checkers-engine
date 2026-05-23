# CCERL Baseline Ladder

Protocol: pairs=`all`, positions=`full`, max plies=`240`, neural sims=`768`, draws=`half`.

## Elo

| Rank | Baseline | Elo | CI95 | Games | W-D-L | Score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `iter082` | 1506.1 | 1444.1..1568.0 | 2580 | 2019-174-387 | 0.816 |
| 2 | `iter063` | 1504.2 | 1442.2..1566.2 | 2580 | 1988-229-363 | 0.815 |
| 3 | `iter064` | 1485.0 | 1423.2..1546.9 | 2580 | 1950-232-398 | 0.801 |
| 4 | `iter061` | 1462.6 | 1400.9..1524.3 | 2580 | 1915-215-450 | 0.784 |
| 5 | `iter062` | 1431.1 | 1369.5..1492.7 | 2580 | 1851-219-510 | 0.760 |
| 6 | `iter060` | 1424.6 | 1363.0..1486.1 | 2580 | 1835-225-520 | 0.755 |
| 7 | `iter057` | 1405.2 | 1343.7..1466.7 | 2580 | 1785-248-547 | 0.740 |
| 8 | `iter046` | 1112.0 | 1051.9..1172.1 | 2580 | 1308-145-1127 | 0.535 |
| 9 | `iter029` | 1076.8 | 1016.9..1136.7 | 2580 | 1257-139-1184 | 0.514 |
| 10 | `tt-pvs` | 795.0 | 738.0..852.1 | 2580 | 880-87-1613 | 0.358 |
| 11 | `iter005` | 703.8 | 647.6..760.0 | 2580 | 780-24-1776 | 0.307 |
| 12 | `iter009` | 672.4 | 616.4..728.4 | 2580 | 723-47-1810 | 0.289 |
| 13 | `converter` | 533.5 | 478.6..588.4 | 2580 | 541-16-2023 | 0.213 |
| 14 | `greedy` | 457.0 | 402.7..511.3 | 2580 | 436-23-2121 | 0.173 |
| 15 | `traffic-greedy` | 340.8 | 287.7..393.9 | 2580 | 302-18-2260 | 0.121 |
| 16 | `random` | 0.0 | 0.0..0.0 | 2580 | 0-99-2481 | 0.019 |

## Pairs

| A | B | Score A | W-D-L | Games | Reasons |
|---|---|---:|---:|---:|---|
| `greedy` | `random` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:4, anti_block_goal_full:168 |
| `traffic-greedy` | `random` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:17, anti_block_goal_full:155 |
| `traffic-greedy` | `greedy` | 0.302 | 50-4-118 | 172 | all_pieces_in_goal:94, anti_block_goal_full:74, equal_turn_goal_draw:4 |
| `converter` | `random` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:4, anti_block_goal_full:168 |
| `converter` | `greedy` | 0.634 | 106-6-60 | 172 | all_pieces_in_goal:160, anti_block_goal_full:6, equal_turn_goal_draw:6 |
| `converter` | `traffic-greedy` | 0.773 | 133-0-39 | 172 | all_pieces_in_goal:107, anti_block_goal_full:65 |
| `tt-pvs` | `random` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:3, anti_block_goal_full:169 |
| `tt-pvs` | `greedy` | 0.849 | 144-4-24 | 172 | all_pieces_in_goal:131, anti_block_goal_full:37, equal_turn_goal_draw:4 |
| `tt-pvs` | `traffic-greedy` | 0.930 | 157-6-9 | 172 | all_pieces_in_goal:46, anti_block_goal_full:120, max_ply:6 |
| `tt-pvs` | `converter` | 0.828 | 142-1-29 | 172 | all_pieces_in_goal:152, anti_block_goal_full:19, equal_turn_goal_draw:1 |
| `iter005` | `random` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:6, anti_block_goal_full:166 |
| `iter005` | `greedy` | 0.872 | 149-2-21 | 172 | all_pieces_in_goal:155, anti_block_goal_full:15, equal_turn_goal_draw:2 |
| `iter005` | `traffic-greedy` | 0.904 | 155-1-16 | 172 | all_pieces_in_goal:90, anti_block_goal_full:81, equal_turn_goal_draw:1 |
| `iter005` | `converter` | 0.738 | 126-2-44 | 172 | all_pieces_in_goal:159, anti_block_goal_full:11, equal_turn_goal_draw:2 |
| `iter005` | `tt-pvs` | 0.331 | 56-2-114 | 172 | all_pieces_in_goal:136, anti_block_goal_full:34, equal_turn_goal_draw:2 |
| `iter009` | `random` | 0.991 | 169-3-0 | 172 | all_pieces_in_goal:3, anti_block_goal_full:166, max_ply:3 |
| `iter009` | `greedy` | 0.770 | 131-3-38 | 172 | all_pieces_in_goal:153, anti_block_goal_full:16, equal_turn_goal_draw:2, repetition:1 |
| `iter009` | `traffic-greedy` | 0.907 | 155-2-15 | 172 | all_pieces_in_goal:92, anti_block_goal_full:78, equal_turn_goal_draw:2 |
| `iter009` | `converter` | 0.759 | 129-3-40 | 172 | all_pieces_in_goal:167, anti_block_goal_full:2, equal_turn_goal_draw:3 |
| `iter009` | `tt-pvs` | 0.346 | 57-5-110 | 172 | all_pieces_in_goal:152, anti_block_goal_full:15, equal_turn_goal_draw:4, max_ply:1 |
| `iter009` | `iter005` | 0.474 | 77-9-86 | 172 | all_pieces_in_goal:160, anti_block_goal_full:3, equal_turn_goal_draw:9 |
| `iter029` | `random` | 0.959 | 158-14-0 | 172 | all_pieces_in_goal:6, anti_block_goal_full:152, max_ply:14 |
| `iter029` | `greedy` | 0.994 | 171-0-1 | 172 | all_pieces_in_goal:156, anti_block_goal_full:16 |
| `iter029` | `traffic-greedy` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:115, anti_block_goal_full:57 |
| `iter029` | `converter` | 0.977 | 168-0-4 | 172 | all_pieces_in_goal:167, anti_block_goal_full:5 |
| `iter029` | `tt-pvs` | 0.924 | 154-10-8 | 172 | all_pieces_in_goal:156, anti_block_goal_full:6, equal_turn_goal_draw:4, max_ply:6 |
| `iter029` | `iter005` | 0.988 | 170-0-2 | 172 | all_pieces_in_goal:158, anti_block_goal_full:14 |
| `iter029` | `iter009` | 0.988 | 168-4-0 | 172 | all_pieces_in_goal:132, anti_block_goal_full:36, equal_turn_goal_draw:1, max_ply:1, repetition:2 |
| `iter046` | `random` | 0.971 | 162-10-0 | 172 | all_pieces_in_goal:5, anti_block_goal_full:157, max_ply:10 |
| `iter046` | `greedy` | 0.988 | 170-0-2 | 172 | all_pieces_in_goal:159, anti_block_goal_full:13 |
| `iter046` | `traffic-greedy` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:94, anti_block_goal_full:78 |
| `iter046` | `converter` | 0.962 | 165-1-6 | 172 | all_pieces_in_goal:167, anti_block_goal_full:4, equal_turn_goal_draw:1 |
| `iter046` | `tt-pvs` | 0.898 | 150-9-13 | 172 | all_pieces_in_goal:147, anti_block_goal_full:16, equal_turn_goal_draw:5, max_ply:4 |
| `iter046` | `iter005` | 0.823 | 141-1-30 | 172 | all_pieces_in_goal:160, anti_block_goal_full:11, equal_turn_goal_draw:1 |
| `iter046` | `iter009` | 0.977 | 168-0-4 | 172 | all_pieces_in_goal:166, anti_block_goal_full:6 |
| `iter046` | `iter029` | 0.660 | 94-39-39 | 172 | all_pieces_in_goal:133, equal_turn_goal_draw:39 |
| `iter057` | `random` | 0.997 | 171-1-0 | 172 | all_pieces_in_goal:5, anti_block_goal_full:166, max_ply:1 |
| `iter057` | `greedy` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:161, anti_block_goal_full:11 |
| `iter057` | `traffic-greedy` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:121, anti_block_goal_full:51 |
| `iter057` | `converter` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:170, anti_block_goal_full:2 |
| `iter057` | `tt-pvs` | 0.983 | 166-6-0 | 172 | all_pieces_in_goal:164, anti_block_goal_full:2, equal_turn_goal_draw:5, max_ply:1 |
| `iter057` | `iter005` | 0.991 | 170-1-1 | 172 | all_pieces_in_goal:130, anti_block_goal_full:41, max_ply:1 |
| `iter057` | `iter009` | 0.994 | 170-2-0 | 172 | all_pieces_in_goal:135, anti_block_goal_full:35, equal_turn_goal_draw:1, max_ply:1 |
| `iter057` | `iter029` | 0.840 | 127-35-10 | 172 | all_pieces_in_goal:137, equal_turn_goal_draw:35 |
| `iter057` | `iter046` | 0.892 | 146-15-11 | 172 | all_pieces_in_goal:157, equal_turn_goal_draw:15 |
| `iter060` | `random` | 0.959 | 158-14-0 | 172 | all_pieces_in_goal:5, anti_block_goal_full:153, max_ply:14 |
| `iter060` | `greedy` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:161, anti_block_goal_full:11 |
| `iter060` | `traffic-greedy` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:107, anti_block_goal_full:65 |
| `iter060` | `converter` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:168, anti_block_goal_full:4 |
| `iter060` | `tt-pvs` | 0.962 | 163-5-4 | 172 | all_pieces_in_goal:165, anti_block_goal_full:2, equal_turn_goal_draw:1, max_ply:4 |
| `iter060` | `iter005` | 0.991 | 170-1-1 | 172 | all_pieces_in_goal:133, anti_block_goal_full:38, equal_turn_goal_draw:1 |
| `iter060` | `iter009` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:166, anti_block_goal_full:6 |
| `iter060` | `iter029` | 0.930 | 156-8-8 | 172 | all_pieces_in_goal:164, equal_turn_goal_draw:8 |
| `iter060` | `iter046` | 0.887 | 149-7-16 | 172 | all_pieces_in_goal:165, equal_turn_goal_draw:7 |
| `iter060` | `iter057` | 0.500 | 75-22-75 | 172 | all_pieces_in_goal:150, equal_turn_goal_draw:22 |
| `iter061` | `random` | 0.948 | 154-18-0 | 172 | all_pieces_in_goal:5, anti_block_goal_full:149, max_ply:18 |
| `iter061` | `greedy` | 0.997 | 171-1-0 | 172 | all_pieces_in_goal:153, anti_block_goal_full:18, equal_turn_goal_draw:1 |
| `iter061` | `traffic-greedy` | 0.997 | 171-1-0 | 172 | all_pieces_in_goal:88, anti_block_goal_full:83, equal_turn_goal_draw:1 |
| `iter061` | `converter` | 0.991 | 170-1-1 | 172 | all_pieces_in_goal:168, anti_block_goal_full:3, equal_turn_goal_draw:1 |
| `iter061` | `tt-pvs` | 0.968 | 164-5-3 | 172 | all_pieces_in_goal:167, max_ply:5 |
| `iter061` | `iter005` | 0.994 | 170-2-0 | 172 | all_pieces_in_goal:135, anti_block_goal_full:35, equal_turn_goal_draw:2 |
| `iter061` | `iter009` | 0.991 | 169-3-0 | 172 | all_pieces_in_goal:165, anti_block_goal_full:4, equal_turn_goal_draw:2, max_ply:1 |
| `iter061` | `iter029` | 0.930 | 158-4-10 | 172 | all_pieces_in_goal:167, anti_block_goal_full:1, equal_turn_goal_draw:4 |
| `iter061` | `iter046` | 0.901 | 152-6-14 | 172 | all_pieces_in_goal:166, equal_turn_goal_draw:6 |
| `iter061` | `iter057` | 0.567 | 88-19-65 | 172 | all_pieces_in_goal:153, equal_turn_goal_draw:19 |
| `iter061` | `iter060` | 0.471 | 58-46-68 | 172 | all_pieces_in_goal:126, equal_turn_goal_draw:46 |
| `iter062` | `random` | 0.965 | 160-12-0 | 172 | all_pieces_in_goal:3, anti_block_goal_full:157, max_ply:12 |
| `iter062` | `greedy` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:164, anti_block_goal_full:8 |
| `iter062` | `traffic-greedy` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:92, anti_block_goal_full:80 |
| `iter062` | `converter` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:169, anti_block_goal_full:3 |
| `iter062` | `tt-pvs` | 0.939 | 153-17-2 | 172 | all_pieces_in_goal:154, anti_block_goal_full:1, equal_turn_goal_draw:5, max_ply:12 |
| `iter062` | `iter005` | 0.991 | 170-1-1 | 172 | all_pieces_in_goal:159, anti_block_goal_full:12, equal_turn_goal_draw:1 |
| `iter062` | `iter009` | 0.994 | 170-2-0 | 172 | all_pieces_in_goal:164, anti_block_goal_full:6, equal_turn_goal_draw:2 |
| `iter062` | `iter029` | 0.907 | 151-10-11 | 172 | all_pieces_in_goal:162, equal_turn_goal_draw:10 |
| `iter062` | `iter046` | 0.826 | 124-36-12 | 172 | all_pieces_in_goal:136, equal_turn_goal_draw:36 |
| `iter062` | `iter057` | 0.558 | 88-16-68 | 172 | all_pieces_in_goal:156, equal_turn_goal_draw:16 |
| `iter062` | `iter060` | 0.584 | 92-17-63 | 172 | all_pieces_in_goal:155, equal_turn_goal_draw:17 |
| `iter062` | `iter061` | 0.390 | 61-12-99 | 172 | all_pieces_in_goal:159, anti_block_goal_full:1, equal_turn_goal_draw:12 |
| `iter063` | `random` | 0.953 | 156-16-0 | 172 | all_pieces_in_goal:5, anti_block_goal_full:151, max_ply:16 |
| `iter063` | `greedy` | 0.997 | 171-1-0 | 172 | all_pieces_in_goal:162, anti_block_goal_full:9, equal_turn_goal_draw:1 |
| `iter063` | `traffic-greedy` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:91, anti_block_goal_full:81 |
| `iter063` | `converter` | 0.994 | 171-0-1 | 172 | all_pieces_in_goal:169, anti_block_goal_full:3 |
| `iter063` | `tt-pvs` | 0.971 | 164-6-2 | 172 | all_pieces_in_goal:166, equal_turn_goal_draw:3, max_ply:3 |
| `iter063` | `iter005` | 0.997 | 171-1-0 | 172 | all_pieces_in_goal:114, anti_block_goal_full:57, equal_turn_goal_draw:1 |
| `iter063` | `iter009` | 0.994 | 170-2-0 | 172 | all_pieces_in_goal:161, anti_block_goal_full:9, equal_turn_goal_draw:2 |
| `iter063` | `iter029` | 0.953 | 161-6-5 | 172 | all_pieces_in_goal:166, equal_turn_goal_draw:6 |
| `iter063` | `iter046` | 0.881 | 147-9-16 | 172 | all_pieces_in_goal:163, equal_turn_goal_draw:9 |
| `iter063` | `iter057` | 0.564 | 63-68-41 | 172 | all_pieces_in_goal:104, equal_turn_goal_draw:68 |
| `iter063` | `iter060` | 0.706 | 110-23-39 | 172 | all_pieces_in_goal:148, anti_block_goal_full:1, equal_turn_goal_draw:23 |
| `iter063` | `iter061` | 0.593 | 81-42-49 | 172 | all_pieces_in_goal:130, equal_turn_goal_draw:42 |
| `iter063` | `iter062` | 0.503 | 77-19-76 | 172 | all_pieces_in_goal:153, equal_turn_goal_draw:19 |
| `iter064` | `random` | 0.974 | 163-9-0 | 172 | all_pieces_in_goal:2, anti_block_goal_full:161, max_ply:9 |
| `iter064` | `greedy` | 0.994 | 170-2-0 | 172 | all_pieces_in_goal:157, anti_block_goal_full:13, equal_turn_goal_draw:2 |
| `iter064` | `traffic-greedy` | 0.985 | 168-3-1 | 172 | all_pieces_in_goal:122, anti_block_goal_full:47, equal_turn_goal_draw:2, max_ply:1 |
| `iter064` | `converter` | 0.974 | 167-1-4 | 172 | all_pieces_in_goal:169, anti_block_goal_full:2, equal_turn_goal_draw:1 |
| `iter064` | `tt-pvs` | 0.951 | 162-3-7 | 172 | all_pieces_in_goal:168, anti_block_goal_full:1, equal_turn_goal_draw:1, max_ply:2 |
| `iter064` | `iter005` | 0.994 | 171-0-1 | 172 | all_pieces_in_goal:165, anti_block_goal_full:7 |
| `iter064` | `iter009` | 0.980 | 166-5-1 | 172 | all_pieces_in_goal:159, anti_block_goal_full:8, equal_turn_goal_draw:5 |
| `iter064` | `iter029` | 0.968 | 165-3-4 | 172 | all_pieces_in_goal:169, equal_turn_goal_draw:3 |
| `iter064` | `iter046` | 0.933 | 158-5-9 | 172 | all_pieces_in_goal:167, equal_turn_goal_draw:5 |
| `iter064` | `iter057` | 0.669 | 91-48-33 | 172 | all_pieces_in_goal:123, anti_block_goal_full:1, equal_turn_goal_draw:48 |
| `iter064` | `iter060` | 0.500 | 66-40-66 | 172 | all_pieces_in_goal:129, anti_block_goal_full:3, equal_turn_goal_draw:40 |
| `iter064` | `iter061` | 0.485 | 64-39-69 | 172 | all_pieces_in_goal:133, equal_turn_goal_draw:39 |
| `iter064` | `iter062` | 0.637 | 89-41-42 | 172 | all_pieces_in_goal:131, equal_turn_goal_draw:41 |
| `iter064` | `iter063` | 0.509 | 79-17-76 | 172 | all_pieces_in_goal:154, anti_block_goal_full:1, equal_turn_goal_draw:17 |
| `iter082` | `random` | 0.994 | 170-2-0 | 172 | all_pieces_in_goal:4, anti_block_goal_full:166, max_ply:2 |
| `iter082` | `greedy` | 1.000 | 172-0-0 | 172 | all_pieces_in_goal:163, anti_block_goal_full:9 |
| `iter082` | `traffic-greedy` | 0.997 | 171-1-0 | 172 | all_pieces_in_goal:100, anti_block_goal_full:71, equal_turn_goal_draw:1 |
| `iter082` | `converter` | 0.991 | 170-1-1 | 172 | all_pieces_in_goal:168, anti_block_goal_full:3, max_ply:1 |
| `iter082` | `tt-pvs` | 0.965 | 162-8-2 | 172 | all_pieces_in_goal:163, anti_block_goal_full:1, equal_turn_goal_draw:5, max_ply:3 |
| `iter082` | `iter005` | 0.997 | 171-1-0 | 172 | all_pieces_in_goal:162, anti_block_goal_full:9, equal_turn_goal_draw:1 |
| `iter082` | `iter009` | 0.988 | 168-4-0 | 172 | all_pieces_in_goal:133, anti_block_goal_full:35, equal_turn_goal_draw:4 |
| `iter082` | `iter029` | 0.930 | 157-6-9 | 172 | all_pieces_in_goal:166, equal_turn_goal_draw:6 |
| `iter082` | `iter046` | 0.933 | 157-7-8 | 172 | all_pieces_in_goal:165, equal_turn_goal_draw:7 |
| `iter082` | `iter057` | 0.741 | 120-15-37 | 172 | all_pieces_in_goal:157, equal_turn_goal_draw:15 |
| `iter082` | `iter060` | 0.645 | 90-42-40 | 172 | all_pieces_in_goal:130, equal_turn_goal_draw:42 |
| `iter082` | `iter061` | 0.529 | 83-16-73 | 172 | all_pieces_in_goal:156, equal_turn_goal_draw:16 |
| `iter082` | `iter062` | 0.616 | 88-36-48 | 172 | all_pieces_in_goal:134, anti_block_goal_full:2, equal_turn_goal_draw:36 |
| `iter082` | `iter063` | 0.375 | 55-19-98 | 172 | all_pieces_in_goal:153, equal_turn_goal_draw:19 |
| `iter082` | `iter064` | 0.541 | 85-16-71 | 172 | all_pieces_in_goal:155, anti_block_goal_full:1, equal_turn_goal_draw:16 |

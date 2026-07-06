# Nested-transformer benchmark

| config                | batch | N   | pix/tok | params(M) | peak(GB) | step(ms) | ex/s | status |
|-----------------------|-------|-----|---------|-----------|----------|----------|------|--------|
| maps.yaml             | 16    | 448 | 1024    | 8.133     | 38.51    | 306.7    | 52.2 | OK     |
| maps.yaml             | 32    | -   | -       | -         | -        | -        | -    | KERNEL |
| maps.yaml             | 64    | -   | -       | -         | -        | -        | -    | ERROR  |
| maps_constant.yaml    | 16    | 448 | 1024    | 0.502     | 83.75    | 522.8    | 30.6 | OK     |
| maps_deep.yaml        | 16    | 448 | 1024    | 15.492    | 73.13    | 560.8    | 28.5 | OK     |
| maps_deep.yaml        | 32    | -   | -       | -         | -        | -        | -    | KERNEL |
| maps_deep.yaml        | 64    | -   | -       | -         | -        | -        | -    | OOM    |
| maps_growth128.yaml   | 16    | 448 | 1024    | 19.684    | 78.68    | 559.8    | 28.6 | OK     |
| maps_growth128.yaml   | 32    | -   | -       | -         | -        | -        | -    | KERNEL |
| maps_window_full.yaml | 64    | -   | -       | -         | -        | -        | -    | KERNEL |

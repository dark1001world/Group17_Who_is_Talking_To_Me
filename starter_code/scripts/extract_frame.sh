#!/bin/bash

mkdir -p data/video_imgs

for file in /DATA/G17/ego4d_data/v2/clips/*.mp4; do
    name=$(basename "$file" .mp4)
    out_dir="data/video_imgs/$name"

    # ✅ Skip if already processed
    if [ -d "$out_dir" ] && [ "$(ls -A $out_dir 2>/dev/null)" ]; then
        echo "Skipping $name (already done)"
        continue
    fi

    echo "Processing $name"
    mkdir -p "$out_dir"

    ffmpeg -y -i "$file" -vf fps=30 -q:v 2 "$out_dir/img_%05d.jpg"
done
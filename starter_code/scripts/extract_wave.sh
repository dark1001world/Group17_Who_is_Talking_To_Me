
for f in `ls /DATA/G17/ego4d_data/v2/v2/clips/`
do
    echo ${f%.*}
    ffmpeg -y -i /DATA/G17/ego4d_data/v2/v2/clips/${f} -qscale:a 0 -ac 1 -vn -threads 6 -ar 16000 /DATA/G17/Data/wave/${f%.*}.wav -loglevel panic
done


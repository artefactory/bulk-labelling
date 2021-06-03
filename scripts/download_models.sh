
echo "Downloading spacy models"
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_md

echo "Downloading tfhub models"
gsutil cp -r gs://bulk-labelling-storage/data/models ./data


from google_play_scraper import reviews
import pandas as pd
import os

# Sentiment word lists (add more words as needed)
positive_words_ar = ['ممتاز', 'رائع', 'جميل', 'مفيد', 'أنصح', 'جيد جدًا', 'أحب', 'مذهل', 'رهيب', 'أفضل']
negative_words_ar = ['سيء', 'فاشل', 'لا يعمل', 'كريه', 'تجربة سيئة', 'مزعج', 'لا أنصح', 'رديء', 'أسوأ', 'خيب أملي']
neutral_words_ar  = ['عادي', 'لا بأس', 'متوسط', 'مقبول', 'جيد فقط', 'ليست الأفضل', 'مش بطال', 'شبه جيد']

positive_words_en = ['excellent', 'awesome', 'good', 'useful', 'recommend', 'love', 'amazing', 'great', 'best', 'perfect']
negative_words_en = ['bad', 'terrible', 'useless', 'hate', 'worst', 'disappointing', 'poor', 'crash', 'buggy', 'fail']
neutral_words_en  = ['okay', 'average', 'mediocre', 'decent', 'not bad', 'just fine', 'so-so', 'acceptable']

# Labeling functions with safe checks
def label_sentiment_ar(text):
    if not text:
        return None
    text = text.strip()
    for word in positive_words_ar:
        if word in text:
            return 'positive'
    for word in negative_words_ar:
        if word in text:
            return 'negative'
    for word in neutral_words_ar:
        if word in text:
            return 'neutral'
    return None

def label_sentiment_en(text):
    if not text:
        return None
    text = text.lower().strip()
    for word in positive_words_en:
        if word in text:
            return 'positive'
    for word in negative_words_en:
        if word in text:
            return 'negative'
    for word in neutral_words_en:
        if word in text:
            return 'neutral'
    return None

# Targets per language and sentiment
target_counts = {
    'ar': {'negative': 2000, 'neutral': 3000},
    'en': {'positive': 1500, 'negative': 3000, 'neutral': 3000}
}

# App package names
apps = [
    'com.twitter.android',          # Twitter
    'com.google.android.youtube',   # YouTube
    'com.ubercab',                  # Uber
    'com.disney.disneyplus'         # Disney+
]

# Output folder
output_folder = "app_review_files"
os.makedirs(output_folder, exist_ok=True)

for app in apps:
    print(f"\n📱 Starting app: {app}")

    collected = {
        'ar': {'negative': [], 'neutral': []},
        'en': {'positive': [], 'negative': [], 'neutral': []}
    }
    seen_texts = set()

    for lang in ['ar', 'en']:
        print(f"🔍 Scraping {lang.upper()} reviews...")

        lang_targets = target_counts.get(lang, {})
        token = None
        page = 0
        max_pages = 100

        while True:
            try:
                result, token = reviews(
                    app,
                    lang=lang,
                    country='SA',
                    count=200,
                    continuation_token=token,
                    filter_score_with=None
                )
            except Exception as e:
                print(f"❌ Error fetching {lang} reviews: {e}")
                break

            if not result:
                print(f"⚠️ No more results returned for {lang.upper()}.")
                break

            new_data_added = False

            for item in result:
                text = item.get('content')
                rating = item.get('score')

                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)

                sentiment = label_sentiment_ar(text) if lang == 'ar' else label_sentiment_en(text)

                if sentiment and sentiment in lang_targets:
                    if len(collected[lang][sentiment]) < lang_targets[sentiment]:
                        collected[lang][sentiment].append({
                            'app': app,
                            'rating': rating,
                            'text': text,
                            'language': lang,
                            'label': sentiment
                        })
                        new_data_added = True

            # Print progress per sentiment
            for sentiment in lang_targets:
                count = len(collected[lang][sentiment])
                goal = lang_targets[sentiment]
                print(f"   → {lang.upper()} {sentiment}: {count}/{goal}")

            # Stop if targets met or no new data added
            if all(len(collected[lang][s]) >= lang_targets[s] for s in lang_targets):
                print(f"✅ Finished {lang.upper()} targets for {app}")
                break
            if not new_data_added:
                print(f"⚠️ No new data found in last page for {lang.upper()} in {app}, stopping.")
                break

            page += 1
            if page >= max_pages:
                print(f"⚠️ Reached max pages for {lang.upper()} in {app}")
                break

    # Combine and save all collected reviews for this app
    all_reviews = []
    for lang in collected:
        for sentiment in collected[lang]:
            all_reviews.extend(collected[lang][sentiment])

    if all_reviews:
        df = pd.DataFrame(all_reviews)
        safe_app_name = app.replace('.', '_')
        file_path = os.path.join(output_folder, f"{safe_app_name}_reviews.csv")
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"💾 Saved {len(df)} reviews to {file_path}")
    else:
        print(f"⚠️ No reviews collected for {app}")

print("\n🎉 All done!")

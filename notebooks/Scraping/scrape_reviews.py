from google_play_scraper import reviews
import pandas as pd

# Arabic sentiment word lists
positive_words_ar = ['ممتاز', 'رائع', 'جميل', 'مفيد', 'أنصح', 'جيد جدًا', 'أحب', 'مذهل', 'رهيب', 'أفضل']
negative_words_ar = ['سيء', 'فاشل', 'لا يعمل', 'كريه', 'تجربة سيئة', 'مزعج', 'لا أنصح', 'رديء', 'أسوأ', 'خيب أملي']
neutral_words_ar  = ['عادي', 'لا بأس', 'متوسط', 'مقبول', 'جيد فقط', 'ليست الأفضل', 'مش بطال', 'شبه جيد']

# English sentiment word lists
positive_words_en = ['excellent', 'awesome', 'good', 'useful', 'recommend', 'love', 'amazing', 'great', 'best', 'perfect']
negative_words_en = ['bad', 'terrible', 'useless', 'hate', 'worst', 'disappointing', 'poor', 'crash', 'buggy', 'fail']
neutral_words_en  = ['okay', 'average', 'mediocre', 'decent', 'not bad', 'just fine', 'so-so', 'acceptable']

# Arabic-only labeling
def label_sentiment_ar(text):
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

# English-only labeling
def label_sentiment_en(text):
    text = text.lower()
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

# Apps to scrape
apps = [
    'com.snapchat.android',
    'com.whatsapp',
    'com.tiktok.z4',
    'com.instagram.android',
    'com.salla.sa',
    'com.hungerstation.android.web',
]

all_reviews = []

# Scrape Arabic and English reviews separately
for app in apps:
    for lang in ['ar', 'en']:
        print(f"Scraping {lang} reviews from: {app}")
        result, _ = reviews(
            app,
            lang=lang,
            country='SA',
            count=3000
        )

        for item in result:
            text = item['content']
            rating = item['score']
            sentiment = label_sentiment_ar(text) if lang == 'ar' else label_sentiment_en(text)

            if sentiment:
                all_reviews.append({
                    'app': app,
                    'rating': rating,
                    'text': text,
                    'language': lang,
                    'label': sentiment
                })

# Convert to DataFrame
df = pd.DataFrame(all_reviews)

# Save all to one Excel file with clear labels, or separate if preferred
df.to_excel('arabic_english_labeled_reviews_latest.xlsx', index=False, engine='openpyxl')
print(f"✅ Saved {len(df)} correctly labeled reviews (Arabic and English separated) to arabic_english_labeled_reviews_latest.xlsx")

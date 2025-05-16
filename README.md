# Facebook Bot with Gemini AI

بوت فيسبوك يستخدم Gemini AI للرد على الرسائل باللهجة الجزائرية.

## المتطلبات

- Python 3.8+
- حساب Railway
- حساب Facebook Developer
- حساب Google Cloud (لـ Gemini API)

## التثبيت

1. قم بنسخ المستودع:
```bash
git clone <repository-url>
cd <repository-name>
```

2. قم بتثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

3. قم بإنشاء ملف `.env` وأضف المتغيرات البيئية التالية:
```
FACEBOOK_PAGE_ACCESS_TOKEN=your_facebook_token
GEMINI_API_KEY_1=your_gemini_key_1
GEMINI_API_KEY_2=your_gemini_key_2
```

## النشر على Railway

1. قم بإنشاء حساب على [Railway](https://railway.app/)
2. قم بربط حساب GitHub الخاص بك
3. قم بإنشاء مشروع جديد واختر "Deploy from GitHub repo"
4. اختر المستودع الخاص بك
5. أضف المتغيرات البيئية في إعدادات المشروع
6. انتظر حتى يتم النشر

## الاستخدام

بعد النشر، قم بإعداد webhook في Facebook Developer Console:
1. انتقل إلى [Facebook Developers](https://developers.facebook.com/)
2. اختر تطبيقك
3. انتقل إلى Messenger > Settings
4. أضف عنوان URL الخاص بـ webhook: `https://your-railway-app-url/webhook`
5. أضف رمز التحقق: `your_verify_token`

## الميزات

- الرد على الرسائل النصية
- معالجة الصور
- دعم اللهجة الجزائرية
- تناوب مفاتيح API
- معالجة الأخطاء
- حفظ البيانات 
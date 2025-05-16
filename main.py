from flask import Flask, request, jsonify
import requests
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
from datetime import datetime, timedelta
import google.generativeai as genai
import asyncio
from typing import Optional, List
import base64
from dotenv import load_dotenv

# تحميل المتغيرات البيئية
load_dotenv()

app = Flask(__name__)

# توكن الوصول والرابط من المتغيرات البيئية
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv('FACEBOOK_PAGE_ACCESS_TOKEN')
FACEBOOK_GRAPH_API_URL = 'https://graph.facebook.com/v11.0/me/messages'
MAX_MESSAGE_LENGTH = 2000

# متغيرات النظامطي
admin = 6793977662  # معرف المسؤول
processed_message_ids = set()
total_users = {}  # نحتفظ فقط بإحصائيات المستخدمين الأساسية
user_context = {}  # نحتفظ فقط بالسياق الأساسي
BOT_START_TIME = None  # وقت بدء تشغيل البوت

# ملفات حفظ البيانات
PROCESSED_IDS_FILE = 'processed_message_ids.pkl'
TOTAL_USERS_FILE = 'total_users.pkl'

# الحد الأقصى للرسائل القديمة لاستردادها عند التشغيل
MAX_HISTORY_MESSAGES = 5
# فترة زمنية لاسترداد الرسائل القديمة (بالساعات)
HISTORY_TIME_WINDOW = 24  # ساعة 

# إضافة استيراد مكتبة Gemini
import google.generativeai as genai
from typing import List
import time

GEMINI_API_KEYS = [
    os.getenv('GEMINI_API_KEY_1'),
    os.getenv('GEMINI_API_KEY_2')
]

class GeminiAPI:
    def __init__(self):
        self.current_key_index = 0
        self.last_error_time = 0
        self.error_cooldown = 5  # Reduced from 60 to 5 seconds for faster rotation
        self.setup_genai()
        print(f"Initialized Gemini API with key index: {self.current_key_index}")
    
    def setup_genai(self):
        genai.configure(api_key=GEMINI_API_KEYS[self.current_key_index])
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.vision_model = genai.GenerativeModel('gemini-2.0-flash')  # Using flash model for both text and vision
    
    def rotate_api_key(self):
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(GEMINI_API_KEYS)
        self.setup_genai()
        print(f"Rotating API key: {old_index} -> {self.current_key_index}")
    
    async def analyze_image(self, image_data: bytes, prompt: str = None) -> Optional[str]:
        attempts = 0
        max_attempts = len(GEMINI_API_KEYS) * 2  # Try each key twice before giving up
        
        while attempts < max_attempts:
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEYS[self.current_key_index]}",
                    headers={
                        'Content-Type': 'application/json'
                    },
                    json={
                        "contents": [{
                            "parts":[
                                {"text": prompt or "Describe this image"},
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": base64.b64encode(image_data).decode('utf-8') if isinstance(image_data, bytes) else image_data
                                    }
                                }
                            ]
                        }]
                    }
                )
                response.raise_for_status()
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                attempts += 1
                error_msg = str(e)
                print(f"Error with API key {self.current_key_index}: {error_msg}")
                
                if "429" in error_msg:  # Rate limit error
                    print(f"Rate limit hit for key {self.current_key_index}, rotating to next key")
                    self.rotate_api_key()
                    continue
                    
                current_time = time.time()
                if current_time - self.last_error_time < self.error_cooldown:
                    print(f"Cooling down key {self.current_key_index} for {self.error_cooldown} seconds")
                    time.sleep(self.error_cooldown)
                
                self.last_error_time = current_time
                self.rotate_api_key()
                
                if attempts >= max_attempts:
                    print("All API keys exhausted after maximum attempts")
                    return "عذرًا، نواجه مشكلة تقنية حاليًا. حاول مرة أخرى بعد قليل."
    
    async def get_response(self, user_message: str) -> Optional[str]:
        attempts = 0
        max_attempts = len(GEMINI_API_KEYS) * 2
        
        while attempts < max_attempts:
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEYS[self.current_key_index]}",
                    headers={
                        'Content-Type': 'application/json'
                    },
                    json={
                        "contents": [{
                            "parts":[{
                                "text": f"""
                                تعليمات المساعد الذكي باللهجة الجزائرية:

                                1. عند السؤال عن المطور/المبرمج:
                                   - "طورني أمين من الجزائر 🇩🇿"
                                   - "تقدر تتواصل معاه على الانستا: amine.kr7"
                                   - لا تذكر أي معلومات تقنية أو نماذج مستخدمة.
                                   - اكتفِ بالقول إنك مصمم من أمين 

                                2. أسلوب التحدث:
                                   - استخدم أسلوباً محايداً تماماً.
                                   - تجنب أي كلمات تشير للجنس.
                                   - لا تستخدم كلمات مثل (خويا، صديقي، عزيزي).
                                   - لا تسأل عن الحال في بداية كل محادثة.
                                   - ركّز على الإجابة المباشرة للسؤال.
                                   - كن محترماً ومهنياً.
                                   - تجنب استخدام الرموز التعبيرية إلا عند الضرورة.
                                   - تجنب استخدام الأحرف المكررة مثل "هههههه".
                                   - استخدم علامات الترقيم بشكل صحيح.
                                   - تجنب استخدام النقاط المتكررة (...).
                                   - تجنب استخدام الأسطر المتكررة.
                                   - اجعل الرد مختصراً وواضحاً.
                                   - تجنب استخدام الكلمات العامية المفرطة.
                                   - استخدم اللغة العربية الفصحى مع بعض الكلمات العامية الجزائرية.
                                   - تجنب استخدام الكلمات الأجنبية إلا عند الضرورة.
                                   - اجعل الرد في سطر واحد أو سطرين كحد أقصى.

                                رسالة المستخدم: {user_message}
                                قم بالرد باللهجة الجزائرية مع مراعاة التعليمات أعلاه.
                                """
                            }]
                        }]
                    }
                )
                response.raise_for_status()
                response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
                
                # تنظيف وتنسيق الرد
                response_text = format_message_for_facebook(response_text)
                
                # التأكد من أن الرد ليس فارغاً
                if not response_text or len(response_text.strip()) < 2:
                    return "عذراً، لم أتمكن من معالجة طلبك. حاول مرة أخرى."
                
                return response_text
                
            except Exception as e:
                attempts += 1
                error_msg = str(e)
                print(f"Error with API key {self.current_key_index}: {error_msg}")
                
                if "429" in error_msg:
                    print(f"Rate limit hit for key {self.current_key_index}, rotating to next key")
                    self.rotate_api_key()
                    continue
                    
                current_time = time.time()
                if current_time - self.last_error_time < self.error_cooldown:
                    print(f"Cooling down key {self.current_key_index} for {self.error_cooldown} seconds")
                    time.sleep(self.error_cooldown)
                
                self.last_error_time = current_time
                self.rotate_api_key()
                
                if attempts >= max_attempts:
                    print("All API keys exhausted after maximum attempts")
                    return "عذرًا، نواجه مشكلة تقنية حاليًا. حاول مرة أخرى بعد قليل."

# Initialize the global Gemini API instance
gemini_api = GeminiAPI()

# تهيئة Gemini API
# GEMINI_API = os.getenv("GEMINI_API", "AIzaSyC8swpbv_LJPo5V3HpF5j94QsAfI633mIs")
# ai_handler = AIHandler(GEMINI_API)

# تحميل البيانات المحفوظة
def load_saved_data():
    """
    تحميل البيانات المحفوظة من الملفات
    """
    global processed_message_ids, user_context, total_users
    
    try:
        if os.path.exists(PROCESSED_IDS_FILE):
            with open(PROCESSED_IDS_FILE, 'rb') as f:
                processed_message_ids = pickle.load(f)
            print(f"تم تحميل {len(processed_message_ids)} معرف رسالة")
            
        if os.path.exists(TOTAL_USERS_FILE):
            with open(TOTAL_USERS_FILE, 'rb') as f:
                total_users = pickle.load(f)
                if isinstance(total_users, set):  # تحويل من set الى dictionary إذا كان قديماً
                    new_total_users = {}
                    for user_id in total_users:
                        new_total_users[user_id] = {
                            'message_count': 0,
                            'first_interaction': datetime.now(),
                            'last_interaction': datetime.now()
                        }
                    total_users = new_total_users
            print(f"تم تحميل معلومات {len(total_users)} مستخدم")
            
    except Exception as e:
        print(f"خطأ في تحميل البيانات: {e}")
        # تهيئة القيم الافتراضية في حالة الخطأ
        processed_message_ids = set()
        total_users = {}

# حفظ البيانات دوريًا
def save_data():
    try:
        with open(PROCESSED_IDS_FILE, 'wb') as f:
            pickle.dump(processed_message_ids, f)
        with open(TOTAL_USERS_FILE, 'wb') as f:
            pickle.dump(total_users, f)
        
        print(f"تم حفظ البيانات: {len(processed_message_ids)} رسالة، {len(total_users)} مستخدم")
    except Exception as e:
        print(f"خطأ في حفظ البيانات: {e}")

def validate_message(message_text):
    """
    التحقق من صحة الرسالة قبل إرسالها
    """
    if not message_text or not isinstance(message_text, str):
        return False
    
    # التحقق من الرسائل المكررة مثل "هههههههه"
    if re.match(r'^(.)\1{10,}$', message_text):
        return False
    
    # التحقق من الأحرف المنقطعة أو غير المفهومة
    if len(message_text.strip()) < 3:
        return False
    
    # التحقق من وجود نص عربي حقيقي في الرسالة
    arabic_text_pattern = re.compile(r'[\u0600-\u06FF\s]{3,}')
    if not arabic_text_pattern.search(message_text):
        return False
    
    return True

def format_message_for_facebook(message_text: str) -> str:
    """
    تنسيق الرسالة بشكل مناسب لفيسبوك
    """
    if not message_text:
        return "عذراً، لم أتمكن من معالجة طلبك. حاول مرة أخرى."
    
    # إزالة الأحرف الخاصة التي قد تسبب مشاكل
    message_text = message_text.replace('\u200b', '')  # إزالة المسافة الصفرية
    message_text = message_text.replace('\u200c', '')  # إزالة المسافة غير المرئية
    message_text = message_text.replace('\u200d', '')  # إزالة الوصل
    message_text = message_text.replace('\u200e', '')  # إزالة LRM
    message_text = message_text.replace('\u200f', '')  # إزالة RLM
    
    # إزالة الأحرف غير المرغوب فيها
    message_text = ''.join(char for char in message_text if ord(char) < 0x10000)
    
    # إزالة النقاط المتكررة
    message_text = re.sub(r'\.{2,}', '.', message_text)
    
    # إزالة المسافات المتكررة
    message_text = re.sub(r'\s+', ' ', message_text)
    
    # إزالة الأسطر المتكررة
    message_text = re.sub(r'\n{2,}', '\n', message_text)
    
    # إزالة علامات الترقيم المتكررة
    message_text = re.sub(r'[!?]{2,}', '!', message_text)
    
    # إزالة الرموز التعبيرية المتكررة
    message_text = re.sub(r'[\U0001F300-\U0001F9FF]{2,}', '😊', message_text)
    
    # التأكد من أن الرسالة لا تبدأ أو تنتهي بمسافات
    message_text = message_text.strip()
    
    # التأكد من أن الرسالة تحتوي على نص عربي
    if not re.search(r'[\u0600-\u06FF]', message_text):
        return "عذراً، لم أتمكن من معالجة طلبك. حاول مرة أخرى."
    
    # التأكد من أن الرسالة ليست فارغة بعد التنظيف
    if not message_text or len(message_text.strip()) < 2:
        return "عذراً، لم أتمكن من معالجة طلبك. حاول مرة أخرى."
    
    # التأكد من أن الرسالة لا تتجاوز الحد الأقصى
    if len(message_text) > MAX_MESSAGE_LENGTH:
        message_text = message_text[:MAX_MESSAGE_LENGTH-100] + "..."
    
    return message_text

# إضافة ثوابت للاتصال
FACEBOOK_API_TIMEOUT = 5  # تقليل وقت الانتظار للاتصال
FACEBOOK_API_RETRY_COUNT = 3  # عدد محاولات إعادة الاتصال
FACEBOOK_API_RETRY_DELAY = 2  # وقت الانتظار بين المحاولات

def make_facebook_request(method, url, **kwargs):
    """
    دالة مساعدة للتعامل مع طلبات Facebook API مع إدارة أفضل للأخطاء
    """
    kwargs.setdefault('timeout', FACEBOOK_API_TIMEOUT)
    kwargs.setdefault('verify', True)  # التأكد من صحة شهادة SSL
    
    for attempt in range(FACEBOOK_API_RETRY_COUNT):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            if attempt < FACEBOOK_API_RETRY_COUNT - 1:
                print(f"محاولة {attempt + 1} فشلت بسبب timeout، جاري إعادة المحاولة...")
                time.sleep(FACEBOOK_API_RETRY_DELAY)
                continue
            raise
        except requests.exceptions.ConnectionError:
            if attempt < FACEBOOK_API_RETRY_COUNT - 1:
                print(f"محاولة {attempt + 1} فشلت بسبب مشكلة في الاتصال، جاري إعادة المحاولة...")
                time.sleep(FACEBOOK_API_RETRY_DELAY)
                continue
            raise
        except requests.exceptions.RequestException as e:
            print(f"خطأ في طلب Facebook API: {str(e)}")
            raise

def send_facebook_message(recipient_id, message_text, quick_replies=None):
    """
    إرسال رسالة إلى مستخدم فيسبوك بشكل محسن
    """
    url = FACEBOOK_GRAPH_API_URL
    params = {
        "access_token": FACEBOOK_PAGE_ACCESS_TOKEN
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    # تنسيق الرسالة
    message_text = format_message_for_facebook(message_text)
    
    data = {
        "recipient": {
            "id": str(recipient_id)
        },
        "message": {
            "text": message_text
        }
    }
    
    if quick_replies:
        data["message"]["quick_replies"] = quick_replies

    try:
        response = make_facebook_request('POST', url, params=params, headers=headers, json=data)
        return True
    except requests.exceptions.RequestException as err:
        if "400" in str(err):
            print(f"خطأ في تنسيق الرسالة للمستخدم {recipient_id}: {message_text[:50]}...")
            try:
                # محاولة إرسال رسالة بديلة
                fallback_data = {
                    "recipient": {"id": str(recipient_id)},
                    "message": {"text": "عذراً، حدث خطأ في معالجة رسالتك. حاول مرة أخرى."}
                }
                make_facebook_request('POST', url, params=params, headers=headers, json=fallback_data)
            except:
                pass
        return False

def notify_admin_of_error(user_id, error_type, error_details):
    """إرسال إشعار الخطأ إلى المسؤول"""
    message = f"🚨 كاين مشكل في البوت:\nUser: {user_id}\nنوع المشكل: {error_type}\nتفاصيل: {error_details}"
    send_facebook_message(admin, message)

# إضافة متغيرات للتحكم في معدل الرسائل
user_message_timestamps = {}  # تخزين توقيت آخر رسالة لكل مستخدم
user_messages_count = {}  # عدد رسائل المستخدم في الدقيقة الأخيرة
user_warnings = {}  # عدد التحذيرات لكل مستخدم
blocked_users = {}  # المستخدمين المحظورين ووقت انتهاء الحظر

# ثوابت للتحكم في معدل الرسائل
MESSAGE_COOLDOWN = 3  # الوقت المطلوب بين الرسائل (بالثواني)
MAX_MESSAGES_PER_MINUTE = 5  # الحد الأقصى للرسائل في الدقيقة
MAX_WARNINGS = 2  # عدد التحذيرات قبل الحظر
BLOCK_DURATION = 300  # مدة الحظر (5 دقائق)

def check_rate_limit(sender_id: str) -> tuple[bool, bool]:
    """
    التحقق من معدل إرسال الرسائل والتحذيرات والحظر
    يعيد (يمكن_الإرسال, تم_الحظر_للتو)
    """
    current_time = datetime.now()

    # التحقق من الحظر
    if sender_id in blocked_users:
        if current_time < blocked_users[sender_id]:
            print(f"المستخدم {sender_id} محظور")
            return False, False
        else:
            # إزالة الحظر والتحذيرات
            del blocked_users[sender_id]
            if sender_id in user_warnings:
                del user_warnings[sender_id]
            if sender_id in user_messages_count:
                del user_messages_count[sender_id]

    # تحديث عدد الرسائل في الدقيقة
    if sender_id not in user_messages_count:
        user_messages_count[sender_id] = {'count': 1, 'reset_time': current_time + timedelta(minutes=1)}
    else:
        if current_time > user_messages_count[sender_id]['reset_time']:
            # إعادة تعيين العداد بعد مرور دقيقة
            user_messages_count[sender_id] = {'count': 1, 'reset_time': current_time + timedelta(minutes=1)}
        else:
            # زيادة عدد الرسائل
            user_messages_count[sender_id]['count'] += 1
            
            # التحقق من تجاوز الحد الأقصى للرسائل
            if user_messages_count[sender_id]['count'] > MAX_MESSAGES_PER_MINUTE:
                user_warnings[sender_id] = user_warnings.get(sender_id, 0) + 1
                if user_warnings[sender_id] >= MAX_WARNINGS:
                    blocked_users[sender_id] = current_time + timedelta(seconds=BLOCK_DURATION)
                    print(f"تم حظر المستخدم {sender_id} لمدة {BLOCK_DURATION//60} دقائق")
                    return False, True
                return False, False

    # التحقق من الوقت بين الرسائل
    if sender_id in user_message_timestamps:
        time_since_last = (current_time - user_message_timestamps[sender_id]).total_seconds()
        if time_since_last < MESSAGE_COOLDOWN:
            user_warnings[sender_id] = user_warnings.get(sender_id, 0) + 1
            if user_warnings[sender_id] >= MAX_WARNINGS:
                blocked_users[sender_id] = current_time + timedelta(seconds=BLOCK_DURATION)
                print(f"تم حظر المستخدم {sender_id} لمدة {BLOCK_DURATION//60} دقائق")
                return False, True
            return False, False

    # تحديث وقت آخر رسالة
    user_message_timestamps[sender_id] = current_time
    return True, False

def handle_facebook_message(sender_id, message_text, message_id, image_data=None):
    """
    معالجة رسالة فيسبوك واردة بشكل محسن
    """
    if message_id in processed_message_ids:
        return
    
    processed_message_ids.add(message_id)
    
    # تحديث إحصائيات المستخدم بشكل مبسط
    if sender_id not in total_users:
        total_users[sender_id] = {'message_count': 0}
    total_users[sender_id]['message_count'] += 1

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if image_data:
            response = loop.run_until_complete(gemini_api.analyze_image(image_data))
        else:
            response = loop.run_until_complete(gemini_api.get_response(message_text))
            
        loop.close()
        
        if response:
            # إرسال الرد مباشرة بدون حفظ بيانات إضافية
            send_facebook_message(sender_id, response)
        
    except Exception as e:
        print(f"خطأ في معالجة الرسالة: {str(e)}")

def poll_facebook_messages():
    """
    دالة مراقبة الرسائل الجديدة بشكل محسن
    """
    consecutive_errors = 0
    max_consecutive_errors = 5
    error_cooldown = 30  # وقت الانتظار بعد عدة أخطاء متتالية
    
    while True:
        try:
            url = f"https://graph.facebook.com/v11.0/me/conversations"
            params = {
                "fields": "messages.limit(5){message,from,id}",
                "access_token": FACEBOOK_PAGE_ACCESS_TOKEN
            }
            
            response = make_facebook_request('GET', url, params=params)
            data = response.json()
            
            # إعادة تعيين عداد الأخطاء المتتالية عند نجاح الطلب
            consecutive_errors = 0
            
            for conversation in data.get('data', []):
                for message in conversation.get('messages', {}).get('data', []):
                    message_id = message.get('id')
                    if not message.get('message') or message_id in processed_message_ids:
                        continue
                    
                    sender_id = message.get('from', {}).get('id')
                    message_text = message.get('message')
                    
                    if sender_id and message_text and sender_id != 'PAGE_ID':
                        handle_facebook_message(
                            sender_id,
                            message_text,
                            message_id
                        )

        except Exception as e:
            consecutive_errors += 1
            print(f"مشكل في قراءة الرسائل: {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                print(f"تم تجاوز الحد الأقصى للأخطاء المتتالية ({max_consecutive_errors})، جاري الانتظار {error_cooldown} ثانية...")
                time.sleep(error_cooldown)
                consecutive_errors = 0
            else:
                time.sleep(FACEBOOK_API_RETRY_DELAY)

        time.sleep(2)

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    """
    معالجة طلبات الويب هوك بشكل محسن
    """
    if request.method == 'GET':
        verify_token = request.args.get('hub.verify_token')
        if verify_token == 'your_verify_token':
            return request.args.get('hub.challenge')
        return 'Invalid verification token'
    
    try:
        data = request.get_json()
        if data['object'] == 'page':
            for entry in data['entry']:
                for messaging_event in entry['messaging']:
                    sender_id = messaging_event['sender']['id']
                    
                    # معالجة الصور
                    if 'message' in messaging_event and 'attachments' in messaging_event['message']:
                        for attachment in messaging_event['message']['attachments']:
                            if attachment['type'] == 'image':
                                try:
                                    image_url = attachment['payload']['url']
                                    image_response = make_facebook_request('GET', image_url)
                                    if image_response.status_code == 200:
                                        handle_facebook_message(
                                            sender_id=sender_id,
                                            message_text="",
                                            message_id=messaging_event['message'].get('mid'),
                                            image_data=image_response.content
                                        )
                                except Exception:
                                    continue
                    
                    # معالجة الرسائل النصية
                    elif 'message' in messaging_event and 'text' in messaging_event['message']:
                        handle_facebook_message(
                            sender_id=sender_id,
                            message_text=messaging_event['message']['text'],
                            message_id=messaging_event['message']['mid']
                        )
    except Exception as e:
        print(f"خطأ في معالجة webhook: {str(e)}")
    
    return jsonify({'status': 'ok'})

# تحديث توكن الفيسبوك وإضافة التحقق من صلاحيته
def verify_facebook_token():
    try:
        url = f"https://graph.facebook.com/v11.0/me"
        response = requests.get(url, params={"access_token": FACEBOOK_PAGE_ACCESS_TOKEN}, timeout=10)
        if response.status_code != 200:
            print(f"Facebook API error: {response.status_code} - {response.text}")
            return False
        return True
    except requests.exceptions.Timeout:
        print("Connection to Facebook timed out. Please check your internet connection.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error verifying token: {e}")
        return False

# تحديث دالة التشغيل الرئيسية
if __name__ == '__main__':
    if not verify_facebook_token():
        print("توكن فيسبوك غير صالح! الرجاء التحقق من التوكن وإعادة التشغيل.")
        exit(1)
    
    BOT_START_TIME = datetime.now()
    print(f"تم بدء تشغيل البوت في: {BOT_START_TIME}")
    
    # تشغيل البوت في خلفية التطبيق
    with ThreadPoolExecutor() as executor:
        executor.submit(poll_facebook_messages)
    
    # تشغيل خادم Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from inverted_index import InvertedIndex # ייבוא מחלקת האינדקס מהקובץ השני
import pickle
from google.cloud import storage

# --- הגדרות סביבה ---
PROJECT_ID = 'YOUR-PROJECT-ID-HERE' # **חובה להחליף במזהה הפרויקט האמיתי שלך**
BASE_DIR = 'index_files'  # התיקייה שבה יישמרו קבצי האינדקס הפיזיים
INDEX_NAME = 'wiki_text_index'
WIKI_XML_FILE = 'wiki_sample.xml' # נניח ששם קובץ ה-XML הוא זה.

# --- 1. פונקציות עזר ל-GCP ---

def get_bucket(bucket_name):
    """מחזיר אובייקט Bucket מ-GCS."""
    if PROJECT_ID == 'YOUR-PROJECT-ID-HERE':
        print("Error: Please replace 'YOUR-PROJECT-ID-HERE' with your actual GCP Project ID.")
        return None
    try:
        return storage.Client(PROJECT_ID).bucket(bucket_name)
    except Exception as e:
        print(f"Error accessing bucket: {e}")
        return None

# --- 2. שלב ה-Map: טוקניזציה ועיבוד טקסט (Normalization) ---

def tokenize_and_normalize(text):
    """
    מבצע טוקניזציה: נירמול (lowercase), הסרת סימני פיסוק ושימוש בביטוי רגולרי
    כדי לחלץ מילים בלבד.
    """
    if not text:
        return []
    
    # 1. Normalization: המרה ל-lowercase
    text = text.lower()
    
    # 2. Tokenization: שימוש בביטוי רגולרי לאחזור מילים (אותיות בלבד) באורך 2 תווים ומעלה
    # זה מונע כניסה של תווים מיוחדים או מילים קצרות מדי (כמו "a" או "I")
    tokens = re.findall(r'\b[a-z]{2,}\b', text)
    
    # כאן ניתן להוסיף Stemming (פורטר) או הסרת Stop Words אם הקוד שלהם קיים.
    
    return tokens

# --- 3. שלב ה-Map: חילוץ נתונים מ-XML (Parsing) ---

def extract_docs_from_xml(xml_path):
    """
    קורא קובץ MediaWiki XML, מחלץ doc_id, כותרת ותוכן גולמי.
    """
    print(f"Reading XML file: {xml_path}")
    docs_data = {}
    
    # הגדרת ה-Namespace של MediaWiki (חיוני לקריאת ה-XML)
    ns = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        return {}

    for page in root.findall('mw:page', ns):
        try:
            doc_id = int(page.find('mw:id', ns).text)
            title = page.find('mw:title', ns).text
            
            # חילוץ תוכן גולמי
            text_element = page.find('mw:revision/mw:text', ns)
            raw_text = text_element.text if text_element is not None else ""
            
            # התעלמות מהפניות (Redirects)
            if page.find('mw:revision/mw:redirect', ns) is not None:
                 continue
                 
            # שילוב הכותרת והתוכן לצורך טוקניזציה
            full_text_to_process = title + " " + raw_text
            docs_data[doc_id] = full_text_to_process
        except Exception as e:
            # במקרה של שגיאה במסמך מסוים, נדפיס הודעה ונמשיך הלאה
            # print(f"Skipping document due to error: {e}")
            continue

    return docs_data

# --- 4. לוגיקת בניית האינדקס הראשית ---

def build_index_job(xml_path, base_dir, index_name, bucket_name=None):
    """
    הלוגיקה הראשית שמבצעת את כל תהליך האינדקסציה.
    """
    
    # 1. יצירת תיקיית הבסיס מקומית אם נדרש
    if bucket_name is None:
        Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    # 2. חילוץ נתונים ועיבוד (Map Phase Simulation)
    raw_docs_data = extract_docs_from_xml(xml_path)
    if not raw_docs_data:
        print("No documents extracted. Aborting index build.")
        return

    print(f"Extracted {len(raw_docs_data)} documents. Starting in-memory indexing...")
    
    # 3. יצירת אובייקט האינדקס
    index = InvertedIndex()
    
    # 4. ביצוע טוקניזציה והוספה לאינדקס (בניית Postings בזיכרון)
    for doc_id, raw_text in raw_docs_data.items():
        # הקריאה לטוקנייזר היא שלב ה-Map
        tokens = tokenize_and_normalize(raw_text)
        
        # הקריאה ל-add_doc היא לוגיקת צבירה (חלק מ-Reduce)
        if tokens:
            index.add_doc(doc_id, tokens)
    
    print("In-memory indexing complete. Vocabulary size:", len(index.df))

    # 5. כתיבת Postings לדיסק (Reduce Phase Output)
    
    # בסימולציה הזו, אנו מחברים את כל ה-Postings לבלוק אחד:
    all_postings = list(index._posting_list.items())
    
    # ⚠️ כתיבת רשימות ה-Postings לדיסק באמצעות הפונקציה הסטטית מהקובץ השני
    # נשתמש ב-'0' כ-bucket_id לסימולציה של בלוק יחיד.
    print(f"Writing {len(all_postings)} posting lists to disk...")
    
    bucket_id = InvertedIndex.write_a_posting_list(
        ('0', all_postings), 
        base_dir, 
        bucket_name
    )
    print(f"Postings for bucket {bucket_id} written successfully.")
    
    # 6. כתיבת המילון (Dictionary) והמיקומים הגלובליים לדיסק
    
    # כדי לשמור את מיקומי הקבצים (posting_locs) שנוצרו
    # אנו צריכים לטעון את קובץ המיקומים שנוצר בשלב 5,
    # ולצרף את המיקומים למילון הראשי.
    
    # טעינת המיקומים שנוצרו בקובץ '0_posting_locs.pickle'
    locations_path = str(Path(base_dir) / f'{bucket_id}_posting_locs.pickle')
    bucket = None if bucket_name is None else get_bucket(bucket_name)
    with InvertedIndex._open(locations_path, 'rb', bucket) as f:
        posting_locs = pickle.load(f)

    # עדכון האינדקס הראשי עם המיקומים שנוצרו
    index.posting_locs.update(posting_locs)
    
    # 7. שמירת האינדקס הגלובלי (df, term_total, posting_locs) לקובץ PKL יחיד
    index.write_index(base_dir, index_name, bucket_name)
    
    print(f"\n--- Indexing Job Complete ---")
    print(f"Total Unique Terms: {len(index.df)}")
    print(f"Index metadata saved to: {base_dir}/{index_name}.pkl")


# --- הפעלת הקוד ---
if __name__ == '__main__':
    # ⚠️ ודא ששני הקבצים נמצאים באותה תיקייה: inverted_index.py ו-index_builder.py
    # ⚠️ ודא שתוכן ה-XML נשמר בקובץ בשם wiki_sample.xml
    
    print("Starting Index Building...")
    
    # דוגמה להרצה מקומית (לא דורש GCS)
    # build_index_job(WIKI_XML_FILE, BASE_DIR, INDEX_NAME)
    
    # דוגמה להרצה עם GCS (יש להחליף את 'your-bucket-name' בשם ה-Bucket שלך)
    # אם תפעיל עם bucket_name, הקבצים יישמרו ב-GCS
    try:
        # אם אתה מריץ מקומית, ודא שה-XML קיים
        if not Path(WIKI_XML_FILE).exists():
             # ניצור קובץ XML קטן עם התוכן ששלחת
             sample_xml_content = """
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/ http://www.mediawiki.org/xml/export-0.10.xsd" version="0.10" xml:lang="en">
  <siteinfo>
    <sitename>Wikipedia</sitename>
    <dbname>enwiki</dbname>
    <base>https://en.wikipedia.org/wiki/Main_Page</base>
    <generator>MediaWiki 1.37.0-wmf.16</generator>
    <case>first-letter</case>
  </siteinfo>
  <page>
    <title>Integration and testing</title>
    <ns>0</ns>
    <id>17324610</id>
    <redirect title="Integration testing" />
    <revision>
      <id>211061398</id>
      <timestamp>2008-05-08T16:50:50Z</timestamp>
      <contributor>
        <username>Kamots</username>
        <id>6021368</id>
      </contributor>
      <comment>[[WP:AES|←]]Redirected page to [[Integration testing]]</comment>
      <model>wikitext</model>
      <format>text/x-wiki</format>
      <text bytes="33" xml:space="preserve">#REDIRECT [[Integration testing]]</text>
      <sha1>n9mmno8pqrdrpndhbjhzpo7ktleyxaa</sha1>
    </revision>
  </page>
  <page>
    <title>Software Development Life Cycle</title>
    <ns>0</ns>
    <id>1000</id>
    <revision>
      <id>999999999</id>
      <timestamp>2023-01-01T00:00:00Z</timestamp>
      <contributor>
        <username>Gemini</username>
        <id>1</id>
      </contributor>
      <model>wikitext</model>
      <format>text/x-wiki</format>
      <text bytes="200" xml:space="preserve">The Software Development Life Cycle (SDLC) is a process used by the software industry to design, develop and test high quality software. The goal is to produce a high-quality product that meets customer expectations.</text>
      <sha1>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx</sha1>
    </revision>
  </page>
</mediawiki>
             """
             with open(WIKI_XML_FILE, 'w', encoding='utf-8') as f:
                 f.write(sample_xml_content.strip())
             print(f"Created sample XML file: {WIKI_XML_FILE}")

        # אם אתה עובד על GCS, הסר את התגובה מהשורה למטה והחלף את שם ה-Bucket
        # GCS_BUCKET_NAME = 'your-bucket-name'
        # build_index_job(WIKI_XML_FILE, BASE_DIR, INDEX_NAME, GCS_BUCKET_NAME)

        # הרצה מקומית (מומלץ להתחיל בזה)
        build_index_job(WIKI_XML_FILE, BASE_DIR, INDEX_NAME)

    except FileNotFoundError:
        print(f"Error: XML file '{WIKI_XML_FILE}' not found.")
    except Exception as e:
        print(f"An error occurred during indexing: {e}")

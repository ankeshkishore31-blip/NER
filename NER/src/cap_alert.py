"""
Common Alerting Protocol (CAP v1.2) Generator and Multilingual Dispatch Service.
Supports Assamese, Bodo, Khasi, Bengali, Hindi, and English.
Adheres to National Disaster Management Authority (NDMA) Sachet specifications.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List

MULTILINGUAL_TEMPLATES = {
    "en": {
        "language_name": "English",
        "headline": "LANDSLIDE EARLY WARNING: High Mass Movement Probability",
        "instruction": "Immediate evacuation from steep cut-slopes to designated community shelters.",
        "description": "Physics-Informed IoT network detected factor of safety breach and pore pressure surge."
    },
    "as": {
        "language_name": "অসমীয়া (Assamese)",
        "headline": "ভূস্খলন সতৰ্কবাৰ্তা: গুৰুতৰ বিপদাশংকা চিহ্নিত",
        "instruction": "তৎক্ষণাৎ বিপজ্জনক পাহাৰীয়া অঞ্চলৰ পৰা নিৰাপদ আশ্ৰয়স্থললৈ স্থানান্তৰিত হওক।",
        "description": "আইঅ'টি চেন্সৰে পাহাৰৰ মাটিৰ তীব্ৰ স্থানচ্যুতি আৰু অত্যাধিক জলচাপ ধৰা পেলাইছে।"
    },
    "brx": {
        "language_name": "बड़ो (Bodo)",
        "headline": "हा खायग्लानाय सांग्रांथि: गिथावना खैफोद",
        "instruction": "थाबैनो हाजो खनाफोरनिफ्राय रैखाथि जायगायाव थां।",
        "description": "मेशिन सिस्टेमा हा खायग्लानायनि गोख्रों सिगनेल मोनदों।"
    },
    "kha": {
        "language_name": "Khasi",
        "headline": "KA JINGMAHAM KHYNDEW TWA: Ka Jingma kaba Khraw",
        "instruction": "Mih noh mardor sha ki jaka ba shngain bad ki Relief Shelter ba la buh kyrpang.",
        "description": "Ki kor sensors ki la lap ba ka khyndew ka lah ban twa mardor ha kane ka thain."
    },
    "bn": {
        "language_name": "বাংলা (Bengali)",
        "headline": "ভূমিধস সতর্কতা: মারাত্মক ঝুঁকির পূর্বাভাস",
        "instruction": "অবিলম্বে পাহাড়ি ঢাল এলাকা ত্যাগ করে নিকটবর্তী আশ্রয়কেন্দ্রে চলে যান।",
        "description": "আইওটি সেন্সর নেটওয়ার্কে তীব্র মাটি স্থানচ্যুতি ও পানির চাপ বৃদ্ধি শনাক্ত হয়েছে।"
    },
    "hi": {
        "language_name": "हिन्दी (Hindi)",
        "headline": "भूस्खलन पूर्व चेतावनी: अत्यधिक खतरे की आशंका",
        "instruction": "पहाड़ी ढलानों से तुरंत सुरक्षित उच्च आश्रय स्थलों पर जाएं।",
        "description": "आईओटी सेंसर नेटवर्क द्वारा ढलान स्थिरता में गंभीर गिरावट दर्ज की गई है।"
    }
}

class AlertManager:
    @staticmethod
    def generate_cap_v12_xml(
        alert_id: str,
        severity: str,
        area_desc: str,
        polygon_coords: str = "25.57,91.89 25.59,91.92 25.55,91.95 25.57,91.89"
    ) -> str:
        """Generates standard OASIS CAP v1.2 XML compliant with NDMA Sachet."""
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        info_blocks = []
        for lang_code, text in MULTILINGUAL_TEMPLATES.items():
            info_blocks.append(f"""    <info>
      <language>{lang_code}-IN</language>
      <category>Geo</category>
      <event>Landslide Hazard Warning</event>
      <urgency>Immediate</urgency>
      <severity>{severity}</severity>
      <certainty>Likely</certainty>
      <headline>{text['headline']}</headline>
      <description>{text['description']}</description>
      <instruction>{text['instruction']}</instruction>
      <area>
        <areaDesc>{area_desc}</areaDesc>
        <polygon>{polygon_coords}</polygon>
      </area>
    </info>""")

        cap_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>{alert_id}</identifier>
  <sender>mdoner.sdma.ner.gov.in</sender>
  <sent>{now_iso}</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
{chr(10).join(info_blocks)}
</alert>"""
        return cap_xml

    @staticmethod
    def get_multilingual_bundle() -> Dict[str, Any]:
        return MULTILINGUAL_TEMPLATES

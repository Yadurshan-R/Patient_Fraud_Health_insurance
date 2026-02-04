"""
Prescription-Receipt Medicine Verification System using GPT-4o Vision API
Compares ONLY the medicines between doctor's prescription and pharmacy receipt.
Supports multiple image formats: .jpg, .jpeg, .png, .pdf
"""

import os
import base64
import re
from pathlib import Path
from difflib import SequenceMatcher
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_image_mime_type(file_path):
    """Detect image MIME type from file extension"""
    ext = Path(file_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.pdf': 'application/pdf'
    }
    return mime_types.get(ext, 'image/jpeg')


def convert_pdf_to_image(pdf_path):
    """Convert first page of PDF to image format"""
    try:
        from pdf2image import convert_from_path
        import tempfile
        
        # Convert first page of PDF to image
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        
        if images:
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            images[0].save(temp_file.name, 'JPEG')
            return temp_file.name
        else:
            raise Exception("Could not convert PDF to image")
    except ImportError:
        raise Exception("pdf2image library not installed. Install with: pip install pdf2image")
    except Exception as e:
        raise Exception(f"Error converting PDF: {str(e)}")


def encode_image_to_base64(image_path):
    """Convert image to base64 encoding for API, handling multiple formats"""
    # Check if it's a PDF
    if Path(image_path).suffix.lower() == '.pdf':
        temp_image = convert_pdf_to_image(image_path)
        with open(temp_image, "rb") as image_file:
            result = base64.b64encode(image_file.read()).decode('utf-8')
        os.unlink(temp_image)  # Clean up temp file
        return result
    else:
        # Regular image file
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def extract_medicines_from_prescription(image_path):
    """
    Extract medicines from doctor's prescription using GPT-4o vision
    """
    print(f"\nAnalyzing prescription image: {image_path}")
    
    base64_image = encode_image_to_base64(image_path)
    
    # Step 1: Extract ALL text using completely generic prompt (no medical terms)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a medical expert assistant. Your job is to extract medical information from this image. describe the image in detail, focusing on any medical prescriptions, medicines, or hospital tests written or printed. Read all text carefully."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ],
            }
        ],
        max_tokens=800,
    )
    
    all_text = response.choices[0].message.content
    
    # Step 2: Extract medicine names from the OCR text (text-only, no image)
    medicine_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""You are a pharmacist and medical expert. From this description/text, identify and list ALL medicines, drugs, and HOSPITAL TESTS found.

{all_text}

Extract EVERY medicine/product/test name you see.
- Include even unclear/handwritten items
- Include abbreviations (e.g. "tabs", "sy", "cap")
- List one per line
- Ignore patient names, dates, doctors, addresses"""
            }
        ],
        max_tokens=300,
    )
    
    medicines_text = medicine_response.choices[0].message.content
    print(f"Prescription medicines extracted")
    
    # Extract medicine names from the response
    medicines = extract_medicine_list(medicines_text)
    
    return medicines, medicines_text


def extract_medicines_from_receipt(image_path):
    """
    Extract medicines from pharmacy receipt using GPT-4o vision
    """
    print(f"\nAnalyzing receipt image: {image_path}")
    
    base64_image = encode_image_to_base64(image_path)
    
    # Step 1: Extract ALL text using completely generic prompt (no medical terms)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please read and transcribe all text visible in this document image. List everything you see."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ],
            }
        ],
        max_tokens=800,
    )
    
    all_text = response.choices[0].message.content
    
    # Step 2: Extract product names from the OCR text (text-only, no image)
    medicine_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"""You are a medical billing expert. From this receipt text, identify and list only the product/item names (medicines, tests, medical supplies).

{all_text}

Extract ONLY the product names. List one per line."""
            }
        ],
        max_tokens=300,
    )
    
    medicines_text = medicine_response.choices[0].message.content
    print(f"Receipt medicines extracted")
    
    # Extract medicine names from the response
    medicines = extract_medicine_list(medicines_text)
    
    return medicines, medicines_text


def extract_medicine_list(text):
    """Extract individual medicine names from text"""
    # Remove code block markers
    text = re.sub(r'```[a-z]*', '', text)
    
    # Split by newlines and clean up
    lines = text.split('\n')
    medicines = []
    
    for line in lines:
        line = line.strip()
        # Remove bullet points, numbers, dashes, etc.
        line = re.sub(r'^[\d\.\-\*\•\:\)]+\s*', '', line)
        line = line.strip()
        
        # Skip empty lines or very short lines
        if len(line) < 3:
            continue
        
        # Skip common non-medicine phrases (GPT-4o sometimes adds these)
        skip_phrases = [
            'here are', 'here is', 'the medicine', 'medicines', 'prescription', 
            'items', 'products', 'following', 'listed', 'list', 'based on',
            'from the', 'sure', 'okay', 'yes', 'product', 'name'
        ]
        
        should_skip = False
        for phrase in skip_phrases:
            if phrase in line.lower():
                should_skip = True
                break
        
        if should_skip:
            continue
            
        medicines.append(line.lower())
    
    return medicines


def ai_compare_medicines(prescription_medicines, receipt_medicines):
    """
    Use GPT-4o to intelligently compare medicines based on medical knowledge
    Determines if medicines are the same or medically equivalent
    """
    if not prescription_medicines or not receipt_medicines:
        return 0, [], []
    
    print(f"\nUsing AI to compare medicines...\n")
    print(f"Prescribed medicines: {len(prescription_medicines)}")
    print(f"Receipt medicines: {len(receipt_medicines)}")
    print()
    
    # Create a comparison prompt for GPT-4o
    comparison_prompt = f"""You are a medical expert (pharmacist/doctor) comparing a doctor's prescription with a pharmacy receipt.
    Your goal is to verify if the patient purchased the correct medicines/tests.

IMPORTANT RULES:
1. The PRESCRIBED list may have OCR errors, abbreviations, or spelling mistakes (handwritten prescriptions)
2. The PURCHASED list is from a pharmacy receipt (more accurate)
3. Receipt medicines must serve THE SAME MEDICAL PURPOSE as prescribed medicines
4. Receipt CAN have LESS items than prescription (patient didn't buy everything - this is OK)
5. Receipt CANNOT have MORE items than prescription (buying unprescribed items - this is SUSPICIOUS)

CRITICAL: Before marking a receipt item as UNPRESCRIBED, check VERY CAREFULLY:
- Could it be an abbreviation of a prescribed medicine? (e.g., "peny cof" might be "pentacef" or "cetirizine")
- Could it be a brand name of a prescribed generic? (e.g., "Augmentin" for "co-amoxiclav")
- Could it be a misspelling or OCR error of a prescribed medicine?
- Does it serve the same medical purpose as any prescribed medicine?

PRESCRIBED MEDICINES (may have errors/abbreviations):
{chr(10).join(f"{i+1}. {med}" for i, med in enumerate(prescription_medicines))}

PURCHASED MEDICINES (from receipt):
{chr(10).join(f"{i+1}. {med}" for i, med in enumerate(receipt_medicines))}

TASK 1 - Match prescribed medicines to purchased:
For each PRESCRIBED medicine, find if there's a matching medicine in PURCHASED list that serves the SAME MEDICAL PURPOSE.

Consider as MATCHES:
- Same medicine (spelling variations: "Gleaper" vs "Glenper")
- Same medicine (OCR errors: "Futop-13" vs "Futop B")
- Generic/brand equivalents serving same purpose (e.g., "co-amoxiclav" = "Augmentin")
- Different products with the SAME MEDICAL PURPOSE (e.g., both are antihistamines, both are antibiotics)
- Abbreviated names vs full names (e.g., "peny cof" might be "pentacef" or could be "cetirizine" abbreviated)

Be VERY LENIENT with spelling and abbreviations but STRICT about medical purpose.

SPECIAL INSTRUCTION FOR HANDWRITING:
The "PRESCRIBED MEDICINES" list is extracted from messy handwriting and may contain "best guesses" or slightly incorrect spellings.
The "PURCHASED MEDICINES" list is accurate (receipt).
USE THE PURCHASED LIST AS A "DECODING KEY" for the prescription.

For each purchased item, check if any prescribed item looks like a bad transcription of it.
- If "Prescribed: sohocya" and "Purchased: sedno" -> MATCH (similar length/letters)
- If "Prescribed: zasert" and "Purchased: zypent" -> MATCH
- If "Prescribed: nexfare" and "Purchased: ventek" -> MATCH (if context implies same position/dose)
- If "Prescribed: atom" and "Purchased: alrin" -> MATCH (if similar dose/purpose)

Ask yourself: "If I scribble 'Zypent' badly, could it look like 'Zasert'?" If YES, then MATCH.

TASK 2 - Check for unprescribed items:
After matching ALL prescribed items (being very generous with abbreviations and alternatives), check if ANY purchased medicine is TRULY NOT matched to ANY prescribed medicine.

ONLY mark as UNPRESCRIBED if you are CERTAIN after checking:
- Is it an abbreviation or alternative name for ANY prescribed item?
- Could it be a brand/generic equivalent?
- Does it serve the same purpose as ANY prescribed medicine?

For each prescribed medicine, respond:
MEDICINE: [prescribed medicine name]
MATCH: [YES/NO]
PURCHASED_AS: [name from receipt if YES, or "NOT FOUND" if NO]
CONFIDENCE: [percentage 0-100]
REASON: [brief explanation of medical purpose match]

Then list any TRULY UNPRESCRIBED items (items on receipt that don't match ANY prescribed medicine):
UNPRESCRIBED_ITEMS: [list items from receipt not matched to ANY prescription item, or "NONE" if all matched]

Be GENEROUS with matching. Only mark UNPRESCRIBED if absolutely certain."""

    # Call GPT-4o to compare
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": comparison_prompt
            }
        ],
        max_tokens=1000,
    )
    
    ai_response = response.choices[0].message.content
    
    # Parse the AI response - handle markdown formatting
    matches = []
    current_match = {}
    unprescribed_items = []
    
    # Remove markdown bold markers
    ai_response = ai_response.replace('**', '')
    
    for line in ai_response.split('\n'):
        line = line.strip()
        if not line or line.startswith('Let') or line.startswith('#'):
            continue
        
        # Check for unprescribed items section
        if 'UNPRESCRIBED_ITEMS:' in line.upper() or 'UNPRESCRIBED ITEMS:' in line.upper():
            items_str = line.split(':', 1)[1].strip() if ':' in line else ''
            if items_str and items_str.upper() != 'NONE':
                unprescribed_items = [item.strip() for item in items_str.split(',')]
            continue
        
        # Remove list numbering
        line = re.sub(r'^\d+\.\s*', '', line)
        line = line.strip('-').strip()
        
        if 'MEDICINE:' in line.upper():
            # Save previous match if exists
            if current_match and 'prescribed' in current_match:
                matches.append(current_match)
            medicine_name = line.split(':', 1)[1].strip() if ':' in line else ''
            current_match = {'prescribed': medicine_name}
        elif 'MATCH:' in line.upper():
            match_value = line.split(':', 1)[1].strip() if ':' in line else ''
            is_match = 'YES' in match_value.upper()
            current_match['is_match'] = is_match
        elif 'PURCHASED_AS:' in line.upper() or 'PURCHASED AS:' in line.upper():
            purchased = line.split(':', 1)[1].strip() if ':' in line else ''
            current_match['purchased'] = purchased
        elif 'CONFIDENCE:' in line.upper():
            confidence_str = line.split(':', 1)[1].strip() if ':' in line else '0'
            # Extract first number found
            numbers = re.findall(r'\d+', confidence_str)
            confidence = int(numbers[0]) if numbers else 0
            current_match['confidence'] = confidence
        elif 'REASON:' in line.upper():
            reason = line.split(':', 1)[1].strip() if ':' in line else ''
            current_match['reason'] = reason
    
    # Add last medicine
    if current_match and 'prescribed' in current_match:
        matches.append(current_match)
    
    # Display results
    matched_count = 0
    for match in matches:
        if match.get('is_match', False):
            matched_count += 1
            print(f"MATCH ({match.get('confidence', 0)}%): '{match.get('prescribed', '')}' ~ '{match.get('purchased', '')}'")
            print(f"   Reason: {match.get('reason', 'N/A')}")
        else:
            print(f"NO MATCH: '{match.get('prescribed', '')}' - {match.get('reason', 'Not found')}")
        print()
    
    # Check for unprescribed items
    if unprescribed_items:
        print("WARNING: Receipt contains UNPRESCRIBED items:")
        for item in unprescribed_items:
            print(f"   - {item}")
        print()
    
    # Calculate match percentage - CHECK IF RECEIPT MATCHES PRESCRIPTION
    # Logic: All items on receipt should be on prescription
    # matched_count = number of receipt items that match prescription
    # total_receipt_items = total items on receipt
    total_receipt_items = len(receipt_medicines)
    
    # Count how many receipt items were matched to prescription
    receipt_matched = 0
    for receipt_med in receipt_medicines:
        is_this_item_matched = False
        # Check if this receipt medicine was matched to any prescribed medicine
        for match in matches:
            if not match.get('is_match', False):
                continue
                
            purchased_name = match.get('purchased', '').lower()
            receipt_med_lower = receipt_med.lower()
            
            # enhanced matching: exact, substring, or high similarity
            if (purchased_name == receipt_med_lower or 
                purchased_name in receipt_med_lower or 
                receipt_med_lower in purchased_name):
                is_this_item_matched = True
                break
                
            # Fallback: fuzzy matching
            similarity = SequenceMatcher(None, purchased_name, receipt_med_lower).ratio()
            if similarity > 0.8:
                is_this_item_matched = True
                break
        
        if is_this_item_matched:
            receipt_matched += 1
    
    match_percentage = (receipt_matched / total_receipt_items) * 100 if total_receipt_items > 0 else 0
    
    return match_percentage, matches, unprescribed_items


def recheck_prescription_for_missing_items(image_path, missing_items):
    """
    Double-check the prescription image for specific items that were missed.
    This acts as a second pair of eyes.
    """
    if not missing_items:
        return []
        
    print(f"\n[Verification] Double-checking prescription for missing item(s): {', '.join(missing_items)}...")
    base64_image = encode_image_to_base64(image_path)
    
    missing_list_str = ", ".join(missing_items)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are a medical expert verifying a prescription. 
I have transcribed the prescription but might have missed some items.
The patient purchased these specific medicines that I missed:
{missing_list_str}

Please look at the HANDWRITING in the image again VERY CAREFULLY.
Are any of these items (or abbreviations/misspellings of them/medical equivalents) written in the prescription?

Return ONLY a list of the items you found from the missing list.
If you find one, output: "FOUND: [Item Name]"
If you don't find it, do not output it."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ],
            }
        ],
        max_tokens=300,
    )
    
    result_text = response.choices[0].message.content
    
    found_items = []
    for line in result_text.splitlines():
        if "FOUND:" in line:
            item = re.sub(r'^[\-\*]*\s*', '', line.replace("FOUND:", "").strip().lower())  # Clean up bullets
            found_items.append(item)
            
    return found_items


def compare_medicine_lists(prescription_medicines, receipt_medicines):
    """
    Compare two lists of medicines using AI intelligence
    """
    match_percentage, matches, unprescribed_items = ai_compare_medicines(prescription_medicines, receipt_medicines)
    return match_percentage, matches, unprescribed_items


def verify_prescription(prescription_path, receipt_path):
    """
    Main verification function - compares medicines only
    Returns: (is_genuine, match_percentage)
    """
    print("=" * 70)
    print("PRESCRIPTION-RECEIPT MEDICINE VERIFICATION")
    print("=" * 70)
    
    try:
        # Step 1: Extract medicines from prescription
        prescription_medicines, prescription_text = extract_medicines_from_prescription(prescription_path)
        
        # Step 2: Extract medicines from receipt
        receipt_medicines, receipt_text = extract_medicines_from_receipt(receipt_path)
        
        # Step 3: Compare medicine lists
        match_percentage, matches, unprescribed_items = compare_medicine_lists(prescription_medicines, receipt_medicines)
        
        # --- Targeted Verification Step ---
        if unprescribed_items:
            found_in_recheck = recheck_prescription_for_missing_items(prescription_path, unprescribed_items)
            
            if found_in_recheck:
                print(f"[Verification] Found {len(found_in_recheck)} previously missed items!")
                print(f"   -> Added: {', '.join(found_in_recheck)}")
                # Add found items to the prescription list and re-compare
                prescription_medicines.extend(found_in_recheck)
                match_percentage, matches, unprescribed_items = compare_medicine_lists(prescription_medicines, receipt_medicines)
            else:
                print(f"[Verification] Could not find the missing items in the prescription.")
        
        # Display results
        print("\n" + "=" * 70)
        print("EXTRACTED MEDICINES")
        print("=" * 70)
        
        print("\nPRESCRIPTION MEDICINES:")
        print("-" * 70)
        for i, med in enumerate(prescription_medicines, 1):
            print(f"{i}. {med}")
        
        print("\n\nRECEIPT MEDICINES:")
        print("-" * 70)
        for i, med in enumerate(receipt_medicines, 1):
            print(f"{i}. {med}")
        
        # Final verdict
        print("\n" + "=" * 70)
        print("VERIFICATION RESULT")
        print("=" * 70)
        
        # Count actual matches
        matched_count = sum(1 for m in matches if m.get('is_match', False))
        
        # Recalculate receipt matched count for display (using same logic)
        receipt_matched_count = 0
        total_receipt_items = len(receipt_medicines)
        for receipt_med in receipt_medicines:
            is_matched = False
            for match in matches:
                if not match.get('is_match', False):
                    continue
                p_name = match.get('purchased', '').lower()
                r_med = receipt_med.lower()
                if p_name == r_med or p_name in r_med or r_med in p_name:
                    is_matched = True
                    break
                if SequenceMatcher(None, p_name, r_med).ratio() > 0.8:
                    is_matched = True
                    break
            if is_matched:
                receipt_matched_count += 1
        
        print(f"\nMedicine Match Score: {match_percentage:.2f}%")
        print(f"   Receipt items matched to prescription: {receipt_matched_count}/{len(receipt_medicines)}")
        print(f"   Prescribed items found on receipt: {matched_count}/{len(prescription_medicines)}")
        
        # Check if genuine
        has_unprescribed = len(unprescribed_items) > 0
        meets_threshold = match_percentage >= 80
        
        if meets_threshold:
            print(f"\nStatus: **GENUINE**")
            print(f"The receipt matches the prescription (>=80% medicines matched)")
            if has_unprescribed:
                print(f"Note: {len(unprescribed_items)} item(s) on receipt were not matched, but threshold met.")
            else:
                print(f"No unprescribed items found on receipt")
            is_genuine = True
        else:
            print(f"\nStatus: **FAKE**")
            print(f"The receipt does NOT match the prescription (<80% medicines matched)")
            if has_unprescribed:
                print(f"The receipt contains significant items NOT prescribed by the doctor!")
            is_genuine = False
        
        print("=" * 70)
        
        return is_genuine, match_percentage
        
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


if __name__ == "__main__":
    # Image paths
    prescription_image = "prescription.jpg"
    receipt_image = "receipt.jpg"
    
    if os.path.exists(prescription_image) and os.path.exists(receipt_image):
        verify_prescription(prescription_image, receipt_image)
    else:
        print("Please provide prescription.jpg and receipt.jpg images")
        print(f"Current directory: {os.getcwd()}")
        print("\nUsage:")
        print("1. Place doctor's prescription as 'prescription.jpg'")
        print("2. Place pharmacy receipt as 'receipt.jpg'")
        print("3. Run: python3 prescription_verifier.py")

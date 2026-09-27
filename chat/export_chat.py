import json
import os

transcript_path = r"C:\Users\vajra\.gemini\antigravity-ide\brain\724918d7-c1d6-465a-a3c0-02cf8c6a51b6\.system_generated\logs\transcript.jsonl"
output_path = r"d:\MP\Project\chat\complete_chat.md"

with open(transcript_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    fout.write("# Complete Chat Transcript\n\n")
    
    for line in fin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        step_type = data.get("type")
        source = data.get("source")
        
        if step_type == "USER_INPUT":
            content = data.get("content", "").strip()
            fout.write(f"## 🧑 User Request\n\n```text\n{content}\n```\n\n---\n\n")
        elif step_type == "PLANNER_RESPONSE" and source == "MODEL":
            content = data.get("content", "").strip()
            if content:
                fout.write(f"## 🤖 AI Assistant\n\n{content}\n\n---\n\n")
                
print(f"Exported chat to {output_path}")

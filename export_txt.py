def sec_to_mmss(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def export_to_txt(segments, output_file="output.txt"):
    """
    segments: list of dict
    Each dict must have: speaker, start, end, text
    """

    with open(output_file, "w", encoding="utf-8") as f:
        for seg in segments:
            start = sec_to_mmss(seg["start"])
            end = sec_to_mmss(seg["end"])
            speaker = seg["speaker"]
            text = seg["text"].strip()

            f.write(f"[{start} - {end}] {speaker}: {text}\n")

    print(f"Saved transcript to {output_file}")

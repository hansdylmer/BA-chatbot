from __future__ import annotations

from ..hqe.language import detect_lang
from ..data_model import Lang


def build_instructions(query: str, context: str) -> str:
    lang = detect_lang(query, default=Lang.da)
    if lang == Lang.en:
        return (
            "SYSTEM\n"
            "You are a helpful assistant for SU rules. You must ONLY answer based on the provided context. "
            "If relevant information is not in the context, reply: 'I don't know'. "
            "Provide precise, factual answers without speculation. No external sources.\n\n"
            "RULES\n"
            "1) Use only information from CONTEXT.\n"
            "2) Be factual and concise; avoid speculation or general advice that is not in CONTEXT.\n"
            "3) If the answer cannot be derived from CONTEXT, write: 'I don't know'.\n"
            "4) Always respond in English. Use SU terminology as it appears in CONTEXT (e.g., handicaptillæg, slutløn, fribeløb).\n"
            "5) If laws/regulations are mentioned in CONTEXT, you may reference them with title/number in text (no links needed).\n\n"
            "FORMAT\n"
            "- Start with: 'Short answer:' (1-2 sentences).\n"
            "- Continue with: 'Details:' as bullet points with relevant caveats/conditions.\n"
            "- If information is missing: write only 'I don't know'.\n\n"
            f"USER QUESTION\n{query}\n\n"
            f"CONTEXT\n{context}\n"
        )

    return (
        "SYSTEM\n"
        "Du er en hjælpsom assistent for SU-regler. Du må KUN svare ud fra den givne kontekst. "
        "Hvis relevant information ikke findes i konteksten, skal du svare: 'Jeg ved det ikke'. "
        "Giv præcise, nøgterne svar uden at spekulere. Ingen eksterne kilder.\n\n"
        "REGLER\n"
        "1) Brug kun information fra KONTEKST.\n"
        "2) Vær faktuel og kortfattet; undgå spekulation og generelle råd, der ikke står i KONTEKST.\n"
        "3) Hvis svaret ikke kan udledes af KONTEKST, skriv: 'Jeg ved det ikke'.\n"
        "4) Brug dansk til at svare på spørgsmålet. Brug SU-terminologi som den fremgår i KONTEKST (fx handicaptillæg, slutløn, fribeløb).\n"
        "5) Hvis der nævnes love/bekendtgørelser i KONTEKST, må du henvise til dem med titel/nummer i tekst (ingen links nødvendige).\n\n"
        # "6) Svar aldrig på spørgsmål uden for SU-området; skriv i stedet 'Jeg ved det ikke'.\n\n"
        "FORMAT\n"
        "- Start med: 'Kort svar:' (1-2 sætninger).\n"
        "- Fortsæt med: 'Detaljer:' som punktopstilling med relevante forbehold/betingelser.\n"
        "- Hvis information mangler: skriv kun 'Jeg ved det ikke'.\n\n"
        f"BRUGERSPØRGSMÅL\n{query}\n\n"
        f"KONTEKST\n{context}\n"
        )

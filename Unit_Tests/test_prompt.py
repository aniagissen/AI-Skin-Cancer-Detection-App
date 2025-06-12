from Prompts import build_prompt

def test_prompt_contains_fields():
    result = build_prompt("Ania", 22, "Female", "UK", "Fair", "Melanoma", "Malignant")
    assert "Ania" in result
    assert "Melanoma" in result
    assert "UK" in result

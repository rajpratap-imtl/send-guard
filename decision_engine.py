def make_recommendation(results, analysis_status):

    malware = results["malware_detection"]

    if (
        malware["status"] == "completed"
        and malware["result"]["classification"] == "malicious"
    ):
        return {
            "action": "hold_for_security_review",
            "message": (
                "The malware detector flagged this executable. "
                "Do not send or execute it without security review."
            )
        }

    if analysis_status in ("partial", "failed"):
        return {
            "action": "review_incomplete_analysis",
            "message": (
                "One or more applicable detectors failed. "
                "File safety or authenticity cannot be established."
            )
        }

    document = results["document_tampering_detection"]

    if document["status"] == "completed":
        return {
            "action": "manual_document_review",
            "message": (
                "Review image-origin findings and suspicious regions. "
                "Document authenticity remains undetermined."
            )
        }

    ai = results["ai_image_detection"]

    if ai["status"] == "completed":

        if ai["result"]["prediction"] == "ai_generated":
            return {
                "action": "review_image_context",
                "message": (
                    "The image was classified as AI-generated. "
                    "Synthetic origin alone does not establish malicious intent."
                )
            }

        return {
            "action": "review_image_context",
            "message": (
                "The image was classified as real. "
                "This does not establish authenticity or file safety."
            )
        }

    if malware["status"] == "completed":
        return {
            "action": "no_malware_indication",
            "message": (
                "The malware detector classified this executable as benign. "
                "A benign prediction does not guarantee file safety."
            )
        }

    return {
        "action": "review_required",
        "message": "No conclusive analysis is available."
    }

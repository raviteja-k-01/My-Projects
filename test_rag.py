from app.services.query_service import QueryService

print("START")

qs = QueryService()

result = qs.query("What is modi?")

print("RESULT OBJECT:")
print(result)

print("\nANSWER:")
print(result["answer"])

print("\nDONE")
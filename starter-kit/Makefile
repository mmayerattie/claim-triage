.PHONY: setup start stop test reindex clean logs health
setup:
	docker compose up --build -d --wait --wait-timeout 120
	@$(MAKE) health
start:
	docker compose up -d --wait --wait-timeout 120
stop:
	docker compose down
# Unit/contract tests inside the API image; a scripted fake LLM means no paid calls.
test:
	docker compose build api
	docker compose run --rm --no-deps -e PORTKEY_API_KEY= api python -m pytest tests -q
# Re-embed every historical claim (normally only changed claim files are re-indexed at startup).
reindex:
	docker compose exec api python -m app.rag --force
health:
	curl --fail --silent --show-error --retry 10 --retry-connrefused --retry-delay 2 http://localhost:8000/api/v1/health
	curl --fail --silent --show-error http://localhost:3000 > /dev/null
logs:
	docker compose logs -f
# Stops only this project. Persistent volumes are preserved.
clean:
	docker compose down --remove-orphans

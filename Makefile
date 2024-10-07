default: fmt

.PHONY: fmt
fmt:
	./sbin/check.sh fmt

.PHONY: clean
clean:
	find . -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -name ".DS_Store" -delete
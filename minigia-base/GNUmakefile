NAME = fnalacceleratormodeling/minigia-base

VERSION = 0.2

.PHONY: all build tag_latest release test

all: build

build:
	docker build --tag $(NAME):$(VERSION) $(PWD)

shell: build
	docker run --rm -it -v $(PWD):/host -w /host $(NAME):$(VERSION) /bin/bash -l

tag_latest: build
	docker tag $(NAME):$(VERSION) $(NAME):latest

release: tag_latest
	@if ! docker images $(NAME) | awk '{ print $$2 }' | grep -q -F $(VERSION); then echo "$(NAME) version $(VERSION) is not yet built. Please run 'make build'"; false; fi
	docker push $(NAME)
	@echo "*** Don't forget to create a tag. git tag rel-$(VERSION) && git push origin rel-$(VERSION)"

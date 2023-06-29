.PHONY: clean datasets

SECRET := "Something I personally told you"

clean:
	find datasets -iname "*.xlsx" -exec rm -v {} \;

datasets: $(shell ls -v1 datasets | grep -i '.xlsx.gpg' | sed 's@^@datasets/@')
	@for f in $^; do 						\
		gpg 							\
			--batch						\
			--decrypt					\
			--output "$${f%.gpg}"				\
			--passphrase "$(SECRET)"			\
			--pinentry-mode loopback			\
			 "$${f}";					\
	done

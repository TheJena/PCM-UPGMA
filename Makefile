.PHONY: clean decrypt encrypt

SHELL:=/bin/bash
SECRET ?= "Something I personally told you"

clean:
	@find . -iname "*.gpg" -print0 |			\
	while IFS= read -r -d '' f; do				\
		if test -f "$${f%.gpg}"; then			\
			rm "$${f%.gpg}";			\
			echo "Removed $${f%.gpg}";		\
		fi						\
	done

decrypt:
	@find . -iname "*.gpg" -print0 |			\
	while IFS= read -r -d '' f; do				\
		if ! test -f "$${f%.gpg}"; then			\
			gpg					\
				--batch				\
				--decrypt			\
				--output "$${f%.gpg}"		\
				--passphrase "$(SECRET)"	\
				--pinentry-mode loopback	\
				--quiet				\
				"$${f}";			\
			echo "Created $${f%.gpg}";		\
		fi						\
	done

encrypt:
	echo "Commented out for safety";
encrypt_real:
	#@find . -iname "*.xlsx" -print0 |			\
	#while IFS= read -r -d '' f; do				\
	#	if ! test -f "$${f}.gpg"; then			\
	#		gpg					\
	#			--batch				\
	#			-c			\
	#			--passphrase "$(SECRET)"	\
	#			--quiet				\
	#			"$${f}";			\
	#		echo "Created $${f}.gpg";		\
	#	fi						\
	#done

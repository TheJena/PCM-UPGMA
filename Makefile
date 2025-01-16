.PHONY: black-inplace black-view clean decrypt decrypt-all encrypt encrypt-all encrypt-pdf

SHELL := /bin/bash
SECRET ?= "Something I personally told you"

# Lets not argue about code style :D
# https://github.com/psf/black#the-uncompromising-code-formatter
black-inplace:
	@clear					\
	&& find . -iname "*.py" -print0 	\
	| xargs -0 -I @ -P $(shell nproc)	\
		black   --color			\
			--line-length 79	\
			--target-version py312	\
			@

black-view:
	@clear					\
	&& find . -iname "*.py" -print0 	\
	| xargs -0 -I @ -P $(shell nproc)	\
		black   --color			\
			--diff			\
			--line-length 79	\
			--target-version py312	\
			@

clean:
	@find . -iname "*.pyc"       -exec rm -v  {} \;
	@find . -iname ".DS_Store"   -exec rm -v  {} \;
	@find . -iname "__pycache__" -empty -delete
	@find . -iname "*.gpg" -print0				\
	| while IFS= read -r -d '' f; do			\
		if test -f "$${f%.gpg}"; then			\
			rm "$${f%.gpg}";			\
			echo "Removed $${f%.gpg}";		\
		fi						\
	done

decrypt:
	@if ! test -f "$${FILE}"; then						\
		echo -e "Usage:\n\tmake decrypt FILE=./foo/bar/path.gpg";	\
	else									\
		gpg								\
			--batch							\
			--decrypt						\
			--output "$${FILE%.gpg}"				\
			--passphrase "$(SECRET)"				\
			--pinentry-mode loopback				\
			--quiet							\
			"$${FILE}"						\
		&& echo "Created $${FILE%.gpg}";				\
	fi

decrypt-all:
	@find . -iname "*.gpg" -print0				\
	| while IFS= read -r -d '' f; do			\
		if ! test -f "$${f%.gpg}"; then			\
			gpg					\
				--batch				\
				--decrypt			\
				--output "$${f%.gpg}"		\
				--passphrase "$(SECRET)"	\
				--pinentry-mode loopback	\
				--quiet				\
				"$${f}"				\
			&& echo "Created $${f%.gpg}";		\
		fi						\
	done

encrypt:
	@if ! test -f "$${FILE}"; then						\
		echo -e "Usage:\n\tmake encrypt FILE=./foo/bar/path.ext";	\
	elif echo "$${FILE}" | grep -qs "\.pdf$"; then				\
		echo -e "Usage:\n\tmake encrypt-pdf FILE=./foo/bar/path.pdf";	\
	else									\
		if test -f "$${FILE%.gpg}.gpg"; then				\
			echo "Warning: overwriting FILE='$${FILE%.gpg}.gpg'";	\
			sleep 5;						\
		fi								\
		&& gpg								\
			--batch							\
			--symmetric						\
			--output -						\
			--passphrase "$(SECRET)"				\
			--pinentry-mode loopback				\
			--quiet							\
			"$${FILE}"						\
		> "$${FILE%.gpg}.gpg"						\
		&& echo "Created $${FILE%.gpg}.gpg";				\
	fi

encrypt-all:
	@if test -z "$${EXT}"; then						\
		echo -e 'Usage:\n\tmake encrypt-all EXT=".xlsx"\t # e.g. ';	\
	else									\
		find . -iname "*$${EXT}" -print0				\
		| while IFS= read -r -d '' f; do				\
			if test -f "$${f}.gpg"; then				\
				echo "Warning: overwriting '$${f}.gpg'";	\
				sleep 5;					\
			fi							\
			&& gpg							\
				--batch						\
				--symmetric					\
				--output - 					\
				--passphrase "$(SECRET)"			\
				--pinentry-mode loopback			\
				--quiet						\
				"$${f}"						\
			> "$${f}.gpg"						\
			&& echo "Created $${f}.gpg";				\
		done;								\
	fi

encrypt-pdf:
	@if ! test -f "$${FILE}"; then						\
		echo -e "Usage:\n\tmake encrypt-pdf FILE=./foo/bar/path.pdf";	\
	else									\
		qpdf --linearize --encrypt					\
		       "$(SECRET)"						\
		       "$(shell base64 -w0 /dev/random      | head -c 32)"	\
		       256							\
		     --annotate=y 						\
		     --assemble=n  						\
		     --extract=n  						\
		     --form=n  							\
		     --modify-other=n  						\
		     --modify=none  						\
		     --print=none  						\
		     -- "$${FILE}" "$${FILE}.tmp"				\
		&& mv "$${FILE}.tmp" "$${FILE}"					\
		&& echo "Encrypted $${FILE}";					\
	fi
# macOS	       "$(shell base64 -i  /dev/random -o - | head -c 32)"	\

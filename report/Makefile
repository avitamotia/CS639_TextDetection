# Build ACL-formatted PDF and copy to repository root as report.pdf (Canvas requirement).
# Uses latexmk if installed; otherwise pdflatex + bibtex (works with BasicTeX).

PDFNAME = report-tmp
ROOTPDF = ../report.pdf
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error -jobname=$(PDFNAME)

all: $(ROOTPDF)

$(ROOTPDF): main.tex references.bib acl.sty acl_natbib.bst
	@if ! command -v pdflatex >/dev/null 2>&1; then \
		echo ""; \
		echo "ERROR: pdflatex not found. Install a TeX distribution, then add it to PATH."; \
		echo "  macOS (MacTeX / BasicTeX): export PATH=\"/Library/TeX/texbin:\$$PATH\""; \
		echo "  See README.md section \"Installing LaTeX (macOS)\"."; \
		echo ""; \
		exit 127; \
	fi
	@if command -v latexmk >/dev/null 2>&1; then \
		echo "Using latexmk"; \
		latexmk -pdf $(LATEX_FLAGS) main.tex; \
	else \
		echo "latexmk not found; using pdflatex + bibtex"; \
		pdflatex $(LATEX_FLAGS) main.tex; \
		bibtex $(PDFNAME); \
		pdflatex $(LATEX_FLAGS) main.tex; \
		pdflatex $(LATEX_FLAGS) main.tex; \
	fi
	cp $(PDFNAME).pdf $(ROOTPDF)
	@echo "Built $(ROOTPDF)"

clean:
	rm -f $(PDFNAME).pdf $(PDFNAME).aux $(PDFNAME).bbl $(PDFNAME).blg $(PDFNAME).log $(PDFNAME).out $(PDFNAME).fdb_latexmk $(PDFNAME).fls $(PDFNAME).toc $(PDFNAME).synctex.gz

.PHONY: all clean

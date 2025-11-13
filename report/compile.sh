#!/bin/bash
# Script para compilar el informe LaTeX

echo "==================================="
echo "Compilando informe LaTeX con LNCS"
echo "==================================="
echo ""

# Verificar que pdflatex está instalado
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex no está instalado"
    echo "Instala LaTeX con: sudo apt-get install texlive-latex-base texlive-latex-extra"
    exit 1
fi

# Compilar el documento (dos pasadas para referencias)
echo "Primera pasada..."
pdflatex -interaction=nonstopmode informe.tex > /dev/null 2>&1

echo "Segunda pasada..."
pdflatex -interaction=nonstopmode informe.tex > /dev/null 2>&1

# Verificar que se generó el PDF
if [ -f "informe.pdf" ]; then
    SIZE=$(du -h informe.pdf | cut -f1)
    PAGES=$(pdfinfo informe.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    
    echo ""
    echo "✓ Compilación exitosa!"
    echo "  Archivo: informe.pdf"
    echo "  Tamaño: $SIZE"
    if [ ! -z "$PAGES" ]; then
        echo "  Páginas: $PAGES"
    fi
    echo ""
    
    # Limpiar archivos auxiliares
    echo "Limpiando archivos auxiliares..."
    rm -f *.aux *.log *.out *.toc
    
    echo "✓ Listo!"
else
    echo ""
    echo "✗ Error: No se pudo generar el PDF"
    echo "  Revisa el archivo informe.log para más detalles"
    exit 1
fi

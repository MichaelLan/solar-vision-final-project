# Despliegue con GitHub Actions

El workflow [deploy.yml](../.github/workflows/deploy.yml) ejecuta pruebas y compilaciones en cada pull request a `main`. En cada push a `main` o ejecución manual, repite esas verificaciones y despliega la API Python junto con `frontend/dist` mediante `uv run pywrangler deploy`. Los resultados de ML y `artifacts/model.json` se versionan; el dataset original permanece excluido por `.gitignore`.

## Preparación necesaria

1. Cree un repositorio en GitHub cuya **raíz sea `solar-vision/`** y cuya rama principal sea `main`. Actualmente esta carpeta no tiene un repositorio Git ni remoto configurado. No habilite además Workers Builds para el mismo Worker: este proyecto usa GitHub Actions como único flujo de despliegue.
2. En Cloudflare, habilite Workers y el subdominio `workers.dev` de su cuenta, o prepare un dominio propio. La configuración usa el nombre de Worker `solar-vision` definido en `backend/wrangler.jsonc`.
3. Copie el **Account ID** desde *Workers & Pages → Account Details* de Cloudflare ([ubicación oficial](https://developers.cloudflare.com/fundamentals/account/find-account-and-zone-ids/)). Cree un **API token** limitado a esa cuenta. Según los [permisos oficiales](https://developers.cloudflare.com/workers/authorization/workers/), crear el Worker por primera vez requiere Workers *Admin* a nivel de producto; para desplegar actualizaciones de uno existente basta *Editor* sobre ese Worker. Una ruta o dominio propio puede requerir además *Workers Routes Write* para la zona correspondiente. No agregue permisos de KV, R2 o D1: este proyecto no los usa.
4. En GitHub, abra *Settings → Secrets and variables → Actions* y cree dos **repository secrets**: `CLOUDFLARE_ACCOUNT_ID` y `CLOUDFLARE_API_TOKEN`. No escriba el token en código, archivos `.env`, mensajes de chat ni logs. Cree también la **repository variable obligatoria** `CLOUDFLARE_WORKER_URL` con la URL completa (por ejemplo, `https://solar-vision.<subdominio>.workers.dev`). Puede formar esta URL con el nombre del Worker configurado y el subdominio `workers.dev` que muestra Cloudflare. El workflow requiere la variable antes de desplegar y, después, verifica públicamente salud, información del modelo, una predicción real y la página principal.
5. Genere y confirme los lockfiles `ml/uv.lock`, `backend/uv.lock` y `frontend/pnpm-lock.yaml` con acceso a los registros de paquetes. El backend ya tiene `uv.lock`. El workflow exige los tres: usa `uv sync --locked` y `pnpm install --frozen-lockfile`, y no despliega si falta alguno. Para generar los dos pendientes, ejecute `cd ml && uv lock` y `cd ../frontend && pnpm install`; después confirme ambos archivos. No cree lockfiles vacíos o parciales.

Para publicar este directorio como repositorio nuevo, después de crear el repositorio vacío en GitHub:

```bash
cd solar-vision
git init -b main
git add .
git status --short
git commit -m "Initial Solar Vision project"
git remote add origin https://github.com/USUARIO/REPOSITORIO.git
git push -u origin main
```

Revise `git status` antes del commit: `dataset/` debe estar excluido y nunca debe aparecer un token. El primer push a `main` inicia el workflow de despliegue; hágalo cuando los lockfiles, secretos y URL estén configurados. Si el repositorio ya existe, configure su remoto y rama según corresponda, sin volver a inicializarlo.

## Qué comprueba el workflow

- Prepara Python 3.12 para ML y 3.14 para el backend, y usa `uv sync --locked` para instalar las dependencias fijadas.
- Ejecuta Pytest y Ruff en ML y backend.
- Reempaqueta `artifacts/model.json` y falla si el módulo de pesos versionado difiere del artefacto.
- Instala pnpm 12.21.0, exige su lockfile, ejecuta las pruebas frontend y compila Vite. Comprueba que `dist/index.html` y al menos un archivo JavaScript existan antes de desplegar.
- Regenera las secciones de resultados a partir de `artifacts/metrics.json`, exige que `report/results.typ` versionado coincida con ellas y compila el PDF con Typst.
- En `main`, exige los secretos y `CLOUDFLARE_WORKER_URL`, despliega y compara la predicción pública con las probabilidades de referencia de una imagen real de prueba.

El workflow no entrena de nuevo durante el despliegue: la evaluación académica y los pesos exportados deben proceder del mismo experimento ya incluido en `artifacts/`. No publique cambios del modelo sin regenerar las métricas, figuras, módulo de pesos e informe.

## Estado actual

El workflow está preparado, pero aún no se ha ejecutado en GitHub: no hay repositorio remoto configurado ni secretos de Cloudflare en esta carpeta. Una prueba local de ASGI confirma el backend; no sustituye la prueba de `pywrangler dev` ni una URL pública verificada.

Documentación: [Pywrangler](https://developers.cloudflare.com/workers/languages/python/packages/), [GitHub Actions para Workers](https://developers.cloudflare.com/workers/ci-cd/external-cicd/github-actions/), [secretos de GitHub Actions](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).

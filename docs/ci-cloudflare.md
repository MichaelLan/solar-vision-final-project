# Despliegue con GitHub Actions

El workflow [deploy.yml](../.github/workflows/deploy.yml) ejecuta pruebas y compilaciones en cada pull request a `main`. En cada push a `main` o ejecución manual, repite esas verificaciones y despliega la API Python junto con `frontend/dist` mediante `uv run pywrangler deploy`. Los resultados de ML y `artifacts/model.json` se versionan; el dataset original permanece excluido por `.gitignore`.

## Preparación necesaria

1. El repositorio ya tiene como **raíz `solar-vision/`**, rama `main` y remoto `git@github.com:MichaelLan/solar-vision-final-project.git`. Mantenga esta raíz para que GitHub reconozca `.github/workflows/deploy.yml`. No habilite además Workers Builds para el mismo Worker: este proyecto usa GitHub Actions como único flujo de despliegue.
2. En Cloudflare, habilite Workers y el subdominio `workers.dev` de su cuenta, o prepare un dominio propio. La configuración usa el nombre de Worker `solar-vision` definido en `backend/wrangler.jsonc`.
3. Copie el **Account ID** desde *Workers & Pages → Account Details* de Cloudflare ([ubicación oficial](https://developers.cloudflare.com/fundamentals/account/find-account-and-zone-ids/)). Cree un **API token** limitado a esa cuenta. Según los [permisos oficiales](https://developers.cloudflare.com/workers/authorization/workers/), crear el Worker por primera vez requiere Workers *Admin* a nivel de producto; para desplegar actualizaciones de uno existente basta *Editor* sobre ese Worker. Una ruta o dominio propio puede requerir además *Workers Routes Write* para la zona correspondiente. No agregue permisos de KV, R2 o D1: este proyecto no los usa.
4. En GitHub, abra *Settings → Secrets and variables → Actions* y cree dos **repository secrets**: `CLOUDFLARE_ACCOUNT_ID` y `CLOUDFLARE_API_TOKEN`. No escriba el token en código, archivos `.env`, mensajes de chat ni logs. Cree también la **repository variable obligatoria** `CLOUDFLARE_WORKER_URL` con la URL completa (por ejemplo, `https://solar-vision.<subdominio>.workers.dev`). Puede formar esta URL con el nombre del Worker configurado y el subdominio `workers.dev` que muestra Cloudflare. El workflow requiere la variable antes de desplegar y, después, verifica públicamente salud, información del modelo, una predicción real y la página principal.
5. Conserve los lockfiles versionados `ml/uv.lock`, `backend/uv.lock` y `frontend/pnpm-lock.yaml`. El lockfile de pnpm 12 contiene dos documentos YAML por diseño. El workflow exige los tres y usa `uv sync --locked` y `pnpm install --frozen-lockfile --ignore-scripts`.

Antes del próximo push a `main`, revise `git status`: `dataset/` debe estar excluido y nunca debe aparecer un token. Configure los secretos y la URL antes de esperar un despliegue exitoso.

## Qué comprueba el workflow

- Prepara Python 3.12 para ML y 3.14 para el backend, y usa `uv sync --locked` para instalar las dependencias fijadas.
- Ejecuta Pytest y Ruff en ML y backend.
- Reempaqueta `artifacts/model.json` y falla si el módulo de pesos versionado difiere del artefacto.
- Instala pnpm 12.21.0 sin scripts de instalación, exige su lockfile, ejecuta las pruebas frontend y compila Vite. Comprueba que `dist/index.html` y al menos un archivo JavaScript existan antes de desplegar.
- Regenera las secciones de resultados a partir de `artifacts/metrics.json`, exige que `report/results.typ` versionado coincida con ellas y compila el PDF con Typst.
- En `main`, exige los secretos y `CLOUDFLARE_WORKER_URL`, despliega y compara la predicción pública con las probabilidades de referencia de una imagen real de prueba.

El workflow no entrena de nuevo durante el despliegue: la evaluación académica y los pesos exportados deben proceder del mismo experimento ya incluido en `artifacts/`. No publique cambios del modelo sin regenerar las métricas, figuras, módulo de pesos e informe.

El job `verify` ejecuta las pruebas y la compilación de JavaScript en un runner de GitHub sin los secretos de Cloudflare. Envía `frontend/dist` como artefacto al job `deploy`, que utiliza otro runner y recibe los secretos solo en los pasos de despliegue. `--ignore-scripts` bloquea los scripts de instalación de paquetes, también si pnpm vuelve a instalar al ejecutar las pruebas o el build; estas tareas sí ejecutan código de dependencias. El funcionamiento de esbuild sin su script de instalación aún no está verificado en CI.

Esta separación reduce la exposición de las credenciales, pero no garantiza seguridad total: un frontend comprometido podría quedar en el artefacto publicado. Las acciones de GitHub y Pywrangler/Wrangler también ejecutan su propio código, incluido JavaScript.

## Estado actual

El repositorio remoto existe y recibió el commit inicial. No se pueden consultar desde esta sesión los logs de GitHub Actions del repositorio privado, por lo que el workflow actualizado no está verificado. El backend, sus pruebas y la compilación Typst pasaron localmente. No se conoce el estado de los secretos configurados en GitHub ni hay una URL pública verificada. Una prueba local de ASGI confirma el backend, pero no sustituye la prueba de `pywrangler dev`.

Documentación: [Pywrangler](https://developers.cloudflare.com/workers/languages/python/packages/), [GitHub Actions para Workers](https://developers.cloudflare.com/workers/ci-cd/external-cicd/github-actions/), [secretos de GitHub Actions](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).

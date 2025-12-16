# Templates for the Frontend (Flask/Jinja2)
Usage
- These templates are used by the Flask app configured in [`api/app.py`](api/app.py) (see functions: [`app.index`](api/app.py), [`app.plot`](api/app.py), [`app.update_plots_bundle`](api/app.py), and [`app.update_plots_bundle`](api/app.py)).
- The app expects templates to be available via the Flask template folder (configured as `Path("frontend") / "templates"` in [`api/app.py`](api/app.py)).

Behavior
- Jinja2 HTML templates that the frontend renders and that embed JS clients which:
  - connect to the server via Socket.IO,
  - receive plot update events (emitted by the server),
  - render interactive plots.

Common conventions
- Template names:
  - `frontend.html` — main UI entrypoint rendered by [`app.index`](api/app.py).
  - `plot_<type>.html` — plot templates. `type` corresponds to plot types used by the server (examples: `pulses`, `contour`, `scatter`). [`app.plot`](api/app.py) selects these by `plot_type`.
- Expected template variables:
  - `port` — Flask/Socket.IO port to connect to (string/int).
  - `plot_id` — unique identifier for the plot instance.
- Socket.IO event naming:
  - Server emits events named: `<plot_type>-<plot_id>` (constructed in [`app.update_plots_bundle`](api/app.py) and [`app.updatupdate_plots_bundle_plot`](api/app.py)).
  - Client-side JS in each `plot_<type>.html` should listen on that event name and update the plot accordingly.

Data contract (what the server sends)
- When a bundle is emitted by [`app.update_plots_bundle`](api/app.py):
  - final message shape:
    - `data` — plot data (object / array, depends on plot type)
    - `meta` — metadata (object) containing at least:
      - `plot_id` — plot identifier
      - `plot_type` — type string
- When [`/update_plots_bundle`](api/app.py) is used, the emitted payload has:
  - `data` (information to plot) and
  - `meta` (contains additional information such as `plot_id` and `plot_type`, `title`, axis labels, etc...)

How to add a new plot template
1. Add a new template file named `plot_<newtype>.html` in this templates folder.
2. Ensure the template provides a JS client that:
   - Connects to Socket.IO on the `port` passed into the template.
   - Listens for `${newtype}-${plot_id}` events.
   - Updates the DOM/plot visualization on receiving data.
3. Register the plot in the frontend configuration so the server knows how to route the proper source to it:
   - See plot configuration in [`frontend/components/config_forms.py`](frontend/components/config_forms.py) (this determines `plot_configurations` used by [`app.update_plots_bundle`](api/app.py)).

Debug tips
- If a plot does not update:
  - Confirm the server is emitting an event with the expected name. Search for `socketio.emit` in [`api/backend/utils/plot_helpers.py`](emit_plot_bundle).
  - Confirm `plot_id` and `plot_type` values sent in the payload match what the client listens for.
  - Check browser console for Socket.IO connection errors (CORS, wrong port, etc.).
- Use the `plot` endpoint (`/plot?type=<type>&id=<id>&port=<port>`) to render a single plot template for manual testing (see [`/plot`](api/backend/routes.py)).

Relevant server code
- Main Flask app: [`api/app.py`](api/app.py) — routes and emit logic [`api/backend/routes.py`].
- Plot configuration source: [`frontend/components/config_forms.py`](frontend/components/config_forms.py).

Suggestions and notes
- Keep templates small and focused: presentation + client-side update logic only.
- Use static assets (JS/CSS) from the sibling `static` folder.

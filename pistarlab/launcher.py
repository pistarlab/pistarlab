from pistarlab.services import ServiceContext
from flask import Flask, escape, request, redirect, Response, json, make_response
from flask_cors import CORS

import logging


import logging
import sys

# Setup Logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

app = Flask(__name__)
CORS(app)
# Disable logging for Flask
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
log.disabled = False
app.logger.disabled = False
logging.info("Starting Launcher")
service_ctx = ServiceContext()


page = """
<head>
</head>
<body>
<h3>PISTARLAB Launcher: Control Panel</h3>
    <div>
        __HTML_CONTENT__
    </div>
    <button onClick="disabebuttons();window.location='/service/toggle_auto_restart'">Toggle Auto Restart</button>
     <br/>
     <br/>
    <br/>
    <br/>
    <button onClick="disabebuttons();window.location='/service/restart_all'">Restart All</button>
    <br/>
    <br/>
    <button onClick="disabebuttons();window.location='/service/stop_all'">Stop All</button>
    <br/>
    <br/>
    <button style="color:red;" onClick="disabebuttons();window.location='/service/reset_all_data'">Reset All Data and Restart Services</button>
    <br/>
    <br/>
    <br/>
    
    <br/>
    <br/>
    <div id="status"></div>
</body>
<script>
var refreshMS= 60000;
var loadTime = new Date();
function timeleft(current, previous) {
    var elapsed = new Date() - loadTime;
    return Math.round((elapsed)/1000);   
}

function disabebuttons(){
    document.querySelectorAll('button').forEach(elem => {
    elem.disabled = true;
    document.getElementById("status").innerHTML = "<strong>Please wait: processing request...</strong>"
    });
}

//function updateTimer(){
    //document.getElementById("lastUpdated").innerHTML = 'Last updated ' +  timeleft() + ' seconds ago.'
    //setTimeout(updateTimer, 500)
//}
//updateTimer()
//window.setTimeout(function () {
//  window.location.reload();
//}, refreshMS);
</script>
"""


def get_page(service_info, auto_restart_info):
    page_list = []
    page_list.append("<table><tr><th>Service</th><th>State</th><th>Actions</th><th>Logs</th><th>Links</th></tr>")
    for name, info in service_info.items():
        state = info['state']

        links = []
        for linkname, link in info['links'].items():
            links.append(f"<a target='_blank' href='{link}'>{link}</a>")

        linkshtml = " | \n".join(links)

        page_list.append(f"""<tr><th>{name}</th><td>
        {state}</td><td>
            <button onClick="disabebuttons();window.location='/service/start/{name}'">start</button>
            <button onClick="disabebuttons();window.location='/service/stop/{name}'">stop</button>
            <button onClick="disabebuttons();window.location='/service/restart/{name}'">restart</button>
            </td><td>
            <a onClick="disabebuttons();" target="_blank" href="/service/logs/{name}">logs</a>
            </td><td>{linkshtml}</td></tr>
        """)
    page_list.append("</table>")

    page_list.append("<br/><br/><br/>")
    page_list.append("<div>Auto Restart Enabled: {}</div>".format(auto_restart_info['enabled']))
    page_list.append("<div>Auto Restart Services: {}</div>".format(", ".join(auto_restart_info['services'])))

    return page.replace("__HTML_CONTENT__", "".join(page_list))


def get_status_page():
    auto_restart_info = service_ctx.get_auto_restart_info()
    service_info = service_ctx.get_service_info()
    return get_page(service_info, auto_restart_info)


@app.route("/")
def index():
    return get_status_page()


@app.route('/service/start/<name>')
def service_start(name):
    service_ctx.get_service(name).start()
    return redirect("/", code=302)


@app.route('/service/stop/<name>')
def service_stop(name):
    service_ctx.get_service(name).stop(name)
    return redirect("/", code=302)


@app.route('/service/restart/<name>')
def service_restart(name):
    service_ctx.get_service(name).restart()
    return redirect("/", code=302)


@app.route('/service/stop_all')
def service_stop_all():
    service_ctx.disable_auto_restart()
    service_ctx.clean_up()
    return redirect("/", code=302)


@app.route('/service/restart_all')
def service_retart_all():
    service_ctx.restart_all()
    return redirect("/", code=302)


@app.route('/service/toggle_auto_restart')
def service_toggle_auto_restart():
    service_ctx.toggle_auto_restart()
    return redirect("/", code=302)


@app.route('/service/logs/<name>')
def service_logs(name):
    return Response(service_ctx.get_service(name).get_log(), mimetype='text')


@app.route('/service/reset_all_data')
def reset_all_data():
    from pistarlab import ctx
    try:
        service_ctx.disable_auto_restart()
        service_ctx.clean_up()
        data_context = ctx.get_data_context()
        data_context.reset_all_data()

    except Exception as e:
        logging.error(e)
    try:
        logging.info("Attempting to restart services")
        service_ctx.start_all()
    except Exception as e:
        logging.error(e)
    return redirect("/", code=302)


@app.route("/api/status")
def api_status():
    response = make_response({"data": service_ctx.get_service_info()})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/api/service/<cmd>/<name>")
def api_admin(cmd, name):
    if cmd == "start":
        service_ctx.get_service(name).start()
    elif cmd == "stop":
        service_ctx.get_service(name).stop()
    elif cmd == "logs":
        service_ctx.get_service(name).get_log()
    elif cmd == "info":
        pass

    response = make_response({"data": service_ctx.get_service_info()})
    response.headers['Content-Type'] = 'application/json'
    return response


def main():
    import argparse
    from colorama import init, Fore, Back, Style
    import os
    init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--launcher_host", default="0.0.0.0", help="Control panel UI, host, default is 0.0.0.0 which will make available from other computers on the network.")
    parser.add_argument("--launcher_port", help="port", default="7776")
    parser.add_argument("--redis_port", help="redis port", default="7771")
    parser.add_argument("--streamer_port", help="redis port", default="7778")
    parser.add_argument("--nostart", action="store_true")
    parser.add_argument("--ray_only", action="store_true")
    parser.add_argument("--disable_auto_restart", action="store_true")
    parser.add_argument("--ray_address", help="Ray head node address. Leave empty or set to localhost to start Local Instance ", default="localhost")
    parser.add_argument("--redis_password", help="redis password", default="5241590000000000")
    parser.add_argument("--disable_xvfb", action="store_true", help="Disable Virtual Frame Buffer (XVFB does not work on Windows)")
    parser.add_argument("--skip_ray_start", action="store_true", help="Skip 'ray start ...' ")
    parser.add_argument("--verbose", action="store_true", help="Increase Output Verbosity")
    parser.add_argument("--enable_dev_ui", action="store_true", help="Enables development UI. TODO: Link to Documentation")
    parser.add_argument("--enable_ide", action="store_true", help="Enables Theia IDE (requires nodejs), TODO: Link to Documentation")
    args = parser.parse_args()

    def print_service_status(show_links=True):
        print(f"{Style.RESET_ALL}")
        print("Services:")
        for name, info in service_ctx.get_service_info().items():
            state = info['state']
            links = ", ".join(info['links'].values())
            if show_links:
                print("{:<15} State: {:<12} Links: {:<35}".format(name, state, links))
            else:
                print("{:<15} State: {:<12}".format(name, state))
        print("")

    services_list = ['redis']

    if not args.disable_xvfb or os.name=='nt':
        services_list.append('xvfb')

    if not args.skip_ray_start:
        services_list.append('ray')

    services_list = services_list + ['backend', 'streamer']

    if args.enable_ide:
        services_list.append('theia_ide')

    if args.enable_dev_ui:
        services_list.append('dev_ui')

    if args.ray_only:
        services_list = ['ray']


    service_ctx.set_commandline_args({
        'launcher_port': args.launcher_port,
        'redis_port': args.redis_port,
        'ray_address': args.ray_address,
        'redis_password': args.redis_password,
        'streamer_port':args.streamer_port
        
    })
    service_ctx.prep_services(services_list)

    service_ctx.verbose = args.verbose

    print("Cleaning up any running services")
    service_ctx.clean_up()
    print_service_status(show_links=False)
    print("Clean up complete")

    import signal

    def graceful_exit(signum=None, frame=None):
        print("Shutting Down Services: signum:{}, frame: {}".format(signum, frame))
        service_ctx.auto_restart_enabled = False
        service_ctx.clean_up()
        print_service_status(show_links=False)
        print("Shutdown Complete")
        sys.exit()

    signal.signal(signal.SIGINT, graceful_exit)
    try:
        if not args.nostart:
            print("Starting Services")
            service_ctx.start_all()
            if not args.disable_auto_restart:
                print("Starting Service Monitor Thread")
                service_ctx.start_service_monitor_thread()
        print("STARTUP COMPLETE")
        print("")
        print(f"{Fore.GREEN}============================================")
        print(f" piSTAR Lab Launcher")
        print("")
        print(f"{Style.RESET_ALL} Control Panel:{Fore.GREEN}http://localhost:{args.launcher_port} ")
        print_service_status()
        print("")
        # if "backend" in services_list:
        #     print(f"{Fore.GREEN}============================================")
        #     print("")
        #     if args.enable_dev_ui:
        #         print(f" {Fore.GREEN}piSTAR Lab Dev UI: http://localhost:8080")
        #     else:
        #         print(f" {Fore.GREEN}piSTAR Lab UI: http://localhost:7777")
        #     print("")
        #     print(f"{Fore.GREEN}============================================")
        #     print("")
        #     print("")

        app.run(host=args.launcher_host, port=args.launcher_port, debug=args.debug, use_reloader=args.debug)
    except (Exception, KeyboardInterrupt) as e:
        print(e)
        graceful_exit()
    sys.exit()


if __name__ == "__main__":
    main()

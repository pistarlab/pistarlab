<template>
<div id="main" :class="[{'collapsed' : collapsed}, {'onmobile' : isOnMobile}]" style>
    <b-modal id="task-manager" size="xl" title="Task History" scrollable :hide-footer="true">
        <TaskHome></TaskHome>
        <div class="mb-5"></div>
    </b-modal>
    <b-modal id="data-browser" size="lg" title="File Data Browser" scrollable :hide-footer="true">
        <DataBrowser></DataBrowser>
        <div class="mb-5"></div>
    </b-modal>

    <b-modal id="sessions" size="xl" title="Sessions" scrollable :hide-footer="true">
        <Sessions></Sessions>
        <div class="mb-5"></div>
    </b-modal>

    <b-modal id="readonly" title="You are in read-only mode">
        This is a read-only Demo
    </b-modal>

    <b-modal id="settings" size="lg" title="Settings" scrollable :hide-footer="true">
        <Settings></Settings>
        <div class="mb-5"></div>
    </b-modal>

    <b-modal id="newtask" size="lg" title="New Task" scrollable :hide-footer="true">
        <TaskSpecs></TaskSpecs>
        <div class="mb-5"></div>
    </b-modal>

    <b-modal id="syslogviewer" size="xl" title="System Logs" scrollable :hide-footer="true">
        <LogViewer height="100%" :nocard="true" :logStreamUrl="`${appConfig.API_URL}/api/stream/scoped/backend_run`"> </LogViewer>
        <div class="mb-5"></div>
    </b-modal>

    <b-modal id="profile" title="User" scrollable :hide-footer="true">
        <Profile> </Profile>
        <div class="mb-5"></div>
    </b-modal>

    <div>
        <div class="main">
            <div class="widget">
             <!-- <router-view /> -->
                    <transition name="fade" mode="out-in">
                        <router-view />
                    </transition>
                <!-- <router-view v-slot="{ Component }">
                    <transition>
                        <component :is="Component" />
                    </transition>
                </router-view> -->
            </div>
        </div>

        <b-sidebar id="sidebar-right" title="piSTAR Lab IDE" right bg-variant="dark" text-variant="light" width="100%" no-header>

            <div v-if="ideenabled" class="h-100" style="width:100%">
                <iframe id="ideframe" src="http://localhost:7781" class="h-100" style="width:100%">
                </iframe>
            </div>
            <template #footer="{ hide }">
                <div class="d-flex bg-dark text-light align-items-center px-3 py-1 bottombar">
                    <strong class="ml-auto footerbrand"><img height="24px" src="/pistar_edit_w.png" /> piSTAR IDE <i class="fa fa-laptop-code"></i></strong>
                    <b-button title="Switch to Lab View" size="sm" variant="dark" class="ml-auto" @click="hide();showLab()"><i class="fa fa-flask"></i> Lab</b-button>

                </div>
            </template>
        </b-sidebar>
    </div>

    <sidebar-menu :disableHover="false" :width="width" :menu="menu" :collapsed="collapsed" :theme="selectedTheme" :show-one-child="true" @toggle-collapse="onToggleCollapse">
        <div slot="header">
            <router-link to="/welcome">
                <div v-if="!collapsed" class="logo">
                    <img height="42px" src="/pistar_edit_w.png" />
                </div>

                <div v-if="!collapsed" class="logo">piSTAR Lab</div>
                <div v-if="collapsed" class="logo">
                    <img height="24px" src="/pistar_edit_w.png" />
                </div>
            </router-link>
        </div>

        <div slot="footer">

        </div>
    </sidebar-menu>

    <div v-if="isOnMobile && !collapsed" class="sidebar-overlay" @click="collapsed = true" />

    <b-navbar class="bottombar py-1" type="dark" fixed="bottom" small="true">

        <b-navbar-nav>
            <b-nav-item id="launchbutton" title="New Task" class="mr-1 bottomnav corneritem" v-b-modal.newtask>
                <i class="fa fa-plus"></i>
            </b-nav-item>
            <b-tooltip target="launchbutton" triggers="hover">
                Launch a new Task
            </b-tooltip>
            <b-nav-item class="mr-1" v-b-modal.sessions>
                <i title="Sessions" class="fa fa-cubes"></i> Sessions
            </b-nav-item>

            <b-nav-item class="mr-1" v-b-modal.task-manager>
                <i title="Task Manager" class="fa fa-cogs"></i> Tasks
            </b-nav-item>

        </b-navbar-nav>
        <b-navbar-nav class="ml-auto">
            <b-nav-text class="appname">
                <i class="fa fa-flask"></i> piSTAR Lab
            </b-nav-text>
            <b-nav-text id="readonlymodebanner" v-if="readOnlyMode" v-b-modal.readonly style="font-weight:900;color:yellow">
                [READ-ONLY]
            </b-nav-text>
            <b-tooltip target="readonlymodebanner" triggers="hover">
                This is a READ-ONLY Instance.
            </b-tooltip>
        </b-navbar-nav>
        <b-navbar-nav class="ml-auto">

            <b-nav-item v-b-popover.hover.top="'Browse Data Files'" class="mr-2" v-b-modal.data-browser>
                <i class="fa fa-folder"></i> Files
            </b-nav-item>
            <!-- <b-nav-item title="Switch to IDE View" class="mr-2" to="/ide">
                <i class="fa fa-laptop-code"></i> IDE
            </b-nav-item> -->
            <!-- <b-nav-item id="idebutton" title="Switch to IDE View" class="mr-2" @click="showIDE()">
                <i class="fa fa-laptop-code"></i> IDE
            </b-nav-item>
                 <b-tooltip target="idebutton" triggers="hover">
                    Launch IDE
                </b-tooltip> -->
            <b-nav-item v-b-popover.hover.top="'Show system logs'" class="ml-auto" v-b-modal.syslogviewer>
                <i class="fa fa-terminal"></i> System Logs
            </b-nav-item>
            <b-nav-text class="ml-auto appstatus-disconnected" title="Disconnected from Server" v-if="!connected">
                <i class="fas fa-exclamation-triangle"></i> Disconnected
            </b-nav-text>
            <b-nav-text class="ml-auto appstatus" title="Connected" v-else>
                <i class="fa fa-plug"></i> Connected
            </b-nav-text>
            <b-nav-item title="Not signed in" v-if="!logged_in" v-b-modal.profile>
                <i class="fas fa-user-slash"></i> {{userId}}
            </b-nav-item>
            <b-nav-item title="Signed in" v-else v-b-modal.profile>
                <i class="fa fa-user"></i> {{userId}}
            </b-nav-item>
            <b-nav-item v-b-popover.hover.top="'Settings'" class="ml-auto" v-b-modal.settings>
                <i class="fa fa-cog"></i>
            </b-nav-item>

        </b-navbar-nav>

    </b-navbar>

</div>
</template>

<script>
import axios from "axios";
import TaskHome from "./views/TaskHome.vue";
import DataBrowser from "./views/DataBrowser.vue";
import Sessions from "./views/SessionHome.vue";
import Settings from "./views/Settings.vue";
import TaskSpecs from "./views/TaskSpecs.vue";
import Profile from "./components/Profile.vue";
// import UserInfoStore from "./shared-data"
import {
    appConfig,
    fetchSettings
} from "./app.config";

export default {
    name: "App",
    components: {
        TaskHome,
        DataBrowser,
        Sessions,
        Settings,
        TaskSpecs,
        Profile

    },
    // https://www.w3schools.com/icons/fontawesome5_icons_science.asp
    data() {
        return {
            appConfig,
            logdataoutput: "_",
            jupyterUrl: "NA",
            rayDashUrl: "",
            logVisible: false,
            ideenabled: false,
            menu: [
                {
                    href: "/",
                    title: "Home",
                    icon: "fas fa-home",
                },
                {
                    href: "/agent/home",
                    title: "Agents",
                    icon: "fas fa-robot",
                },
                {
                    href: "/env/home",
                    title: "Environments",
                    icon: "fa fa-gamepad",
                },
                {
                    title: "",
                    header: true,
                },
                {
                    href: "/extension/home",
                    title: "Extension Manager",
                    icon: "fas fa-puzzle-piece",
                },
                {
                    href: "/workspace/home",
                    title: "Workspace",
                    icon: "fas fa-code",
                },
                {
                    href: "/community/profile",
                    title: "Community Hub",
                    icon: "fa fa-users"
                }, {
                    title: "",
                    header: true,
                },
                {
                    href: "/Help",
                    title: "Help",
                    icon: "fa fa-question-circle",
                }
            ],
            collapsed: false,
            themes: [{
                    name: "Default theme",
                    input: "",
                },
                {
                    name: "White theme",
                    input: "white-theme",
                },
            ],
            selectedTheme: "",
            isOnMobile: false,
            es: null,
            logStream: null,
            logdata: [],
            logInit: false,
            ideWindow: null,
            readOnlyMode: false,

            connected: false,
            logged_in: false,
            userId: null

        };
    },
    mounted() {
        this.onResize();
        window.addEventListener("resize", this.onResize);
    },
    computed: {
        //nada
    },
    methods: {
        showLab() {
            // this.ideenabled = false
            // document.body.style.overflow = 'visible';

            //
        },
        showIDE() {
            this.ideenabled = true
            // document.body.style.overflow = 'hidden';

            if (!this.ideWindow) {
                this.ideWindow = window.open('http://localhost:7781', 'xxx');
                console.log(this.ideWindow)
            }
            // myWindow.document.write("<p>This is 'myWindow'</p>");
            // if(doFocus)
            let result = this.ideWindow.focus();
            console.log(result)

            //
        },
        onToggleCollapse(collapsed) {
            console.log(collapsed);
            this.collapsed = collapsed;
        },
        onResize() {
            if (window.innerWidth <= 2600) {
                this.isOnMobile = true;
                this.collapsed = true;
            } else {
                this.isOnMobile = false;
                this.collapsed = false;
            }
        },
        getServerStatus() {
            axios
                .get(`${appConfig.API_URL}/api/status/`)
                .then((response) => {
                    this.connected = true;
                    this.userId = response.data.user_id
                    this.logged_in = response.data.logged_in
                    this.shared.setLoggedIn(this.logged_in)
                    this.shared.state.userId = this.userId

                })
                .catch((error) => {
                    console.log(error)
                    this.userId = null
                    this.connected = false;
                });
        }

    },
    created: function () {

        fetchSettings().then(settings => {
            this.readOnlyMode = settings.sys_config.read_only_mode
        })

        this.getServerStatus()
        this.timer = setInterval(this.getServerStatus, 3000);
    },
    beforeDestroy() {
        clearInterval(this.timer);
    },
    props: {
        width: {
            type: String,
            default: "200px",
        },
    },
};
</script>

<style>
/* @import url("https://fonts.googleapis.com/css?family=Source+Sans+Pro:400,600"); */

@import "./assets/fontawesome/css/all.css";
@import "./assets/styles/dark-bootstrap.css";

body,
html {
    margin: 0;
    padding: 0;
    height: 100%;
}

body {
    font-family: Roboto, Avenir, Helvetica, Arial, sans-serif;
    /* background-color: #eee;
    ;
    color: #3c4858; */
    font-size: .875em;
    font-weight: 300;

}

.h1,
.h2,
.h3,
.h4,
body,
h1,
h2,
h3,
h4,
h5,
h6 {}

h1 {
    font-family: "Lucida Console", "Courier New", monospace;
}

h1 {
    font-size: 2rem;
    font-weight: 500;
}

#main {
    padding-left: 200px;
    transition: 0.3s ease;
}

#main.collapsed {
    padding-left: 50px;
}

#main.onmobile {
    padding-left: 50px;
}

.sidebar-overlay {
    position: fixed;
    width: 100%;
    height: 100%;

    top: 0;
    left: 0;
    /* background-color: #000; */
    opacity: 0.5;
    z-index: 900;
}

th {
    text-transform: uppercase;
    /* font-weight: normal; */
    /* color:#666; */
    color: #4a4a4a;
    font-weight: 400 !important;
    font-size: 12px;
    letter-spacing: 1.2px;
    /* font-family: "SF Pro Display"; */
    border-top: 0px !important;
}

td {
    font-size: .9rem;
}

.main {
    padding: 0px 15px 15px 15px;

}

.logo {
    color: white;
    text-align: center;
    vertical-align: middle;
    height: 50px;
    padding: 10px;
    padding-top: 15px;
}

.svgagent {
    filter: invert(50%) sepia(100%) saturate(0%) hue-rotate(80deg) brightness(100%) contrast(100%) !important;
}

.v-sidebar-menu {
    padding-bottom: 32px;
    /* box-shadow: 2px 0px 6px #999; */
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2),
        0 6px 20px 0 rgba(0, 0, 0, 0.19);
    background-color: #161b22 !important;

}

.vsm--icon {
    /* background-color: transparent !important; */
    /* color: #fff !important; */
    background-color: #161b22 !important;

    height: 50px;
}

.vsm--item {
    /* background-color: #161b22 !important; */
    /* color: #fff !important; */
    height: 50px;
}

.vsm--link.vsm--link_level-1 {
    color: #9f9f9f !important;
}

hr {
    border-color: #666;
}

.vsm--link.vsm--link_level-1:hover {
    color: #fff !important;
    /* background-color: #4285f4 !important; */
    font-weight: 900;
}

.vsm--link_level-1.vsm--link_exact-active {
    color: #fff !important;
    background-color: #4285f4 !important;
}

.vsm--toggle-btn {
    background-color: #161b22 !important;

}

.vsm--mobile-item {
    max-width: 220px !important;
}

pre {
    font-family: Consolas, monospace;
    /* background: #fff; */
    border-radius: 2px;
    padding: 15px;
    line-height: 1.5;
    overflow: auto;
    background: inherit;

    font-size: 0.7em;
    background: linear-gradient(rgba(0, 0, 0, 0.3),
            rgba(0, 0, 0, 0.3));
    color: #FFF;
}

pre {
    white-space: pre-wrap;
    /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;
    /* Mozilla, since 1999 */
    white-space: -pre-wrap;
    /* Opera 4-6 */
    white-space: -o-pre-wrap;
    /* Opera 7 */
    word-wrap: break-word;
    /* Internet Explorer 5.5+ */
}

pre.error {
    background: inherit;
}

.widget {
    padding: 10px 10px;
    padding-bottom: 20px;
    padding-top: 20px;
    margin-bottom: 20px;
    /* // background-color: #343a40; */
    border-radius: 5px;
}

.part {
    padding-top: 40px;
}

.b-table a {
    /* color:#C2DFFF; */
    text-decoration: none;
}

.b-table {
    font-size: 14px;
}

button .default {
    /* background-color: #3f4c6a !important; */
}

.part {
    padding-top: 40px;
}

.b-table a {
    /* color:#C2DFFF; */
    text-decoration: none;
}

/* .b-table a:hover{
  font-weight:bold;
} */

.b-table {
    font-size: 14px;
}

.data_label {
    color: #aaa;
    margin-bottom: 6px
}

.stat_label {
    margin-bottom: 6px;
    /* color: #666; */
    color: #aaa;
}

.stat_value {
    font-size: 1.3em;
    color: #bbb;

}

.image-box {
    height: 200px;
}

.feature-image {
    box-shadow: 0 0 10px #333;
    max-height: 100%;
}

.card {
    border: 0;
    margin-bottom: 30px;
    margin-top: 30px;
    border-radius: 6px;
    /* color: #333; */
    /* background: #fff; */
    width: 100%;
    box-shadow: 0 1px 4px 0 rgba(0, 0, 0, .14);
}

.breadcrumb {
    /* background-color: #fff !important; */

}

.card-flyer {
    border-radius: 5px;

}

.card-flyer .image-box {
    /* background: #000; */
    overflow: hidden;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.20);

    border-radius: 5px;
    height: 200px;
    width: 100%;
}

.card-flyer .image-box img {
    -webkit-transition: all .2s ease;
    -moz-transition: all .2s ease;
    -o-transition: all .2s ease;
    -ms-transition: all .2s ease;

    width: 100%;

}

.card-flyer:hover .image-box img {
    /* opacity: 0.7; */
    -webkit-transform: scale(1.1);
    -moz-transform: scale(1.1);
    -ms-transform: scale(1.1);
    -o-transform: scale(1.1);
    transform: scale(1.1);
}

.card-flyer .text-box {
    text-align: center;
}

.card-flyer .text-box .text-container {
    padding: 30px 18px;
}

.card-flyer {
    /* background: #FFFFFF; */
    margin-top: 0px;
    -webkit-transition: all 0.2s ease-in;
    -moz-transition: all 0.2s ease-in;
    -ms-transition: all 0.2s ease-in;
    -o-transition: all 0.2s ease-in;
    transition: all 0.2s ease-in;
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2),
        0 6px 20px 0 rgba(0, 0, 0, 0.19);

}

.card-flyer:hover {
    /* background: #fff; */
    box-shadow: 0px 15px 26px rgba(0, 0, 0, 0.50);
    -webkit-transition: all 0.2s ease-in;
    -moz-transition: all 0.2s ease-in;
    -ms-transition: all 0.2s ease-in;
    -o-transition: all 0.2s ease-in;
    transition: all 0.2s ease-in;
    margin-top: 0px;
}

.card-flyer .text-box p {
    margin-top: 10px;
    margin-bottom: 0px;
    padding-bottom: 0px;
    font-size: 14px;
    letter-spacing: 1px;
    /* color: #000000; */
}

.card-flyer .text-box h6 {
    margin-top: 0px;
    margin-bottom: 4px;
    font-size: 18px;
    font-weight: bold;
    text-transform: uppercase;
    font-family: 'Roboto Black', sans-serif;
    letter-spacing: 1px;
}

.footerbrand {
    color: #FFF;
    margin: 3px;
    font-weight: 900;
}

.custom-card-header {
    font-size: 1.5em;
}

.table {
    font-size: .875rem !important;

    /* color: #3c4858d8 !important; */
}

.badge-tag {
    background-color: #ddd;
    color: #000;
}

.navbar-text {
    padding: 0px 0px;
    margin: 5px 5px;
}

.navbar-text {
    padding: 0px 0px;
    margin: 5px 5px;
}

.bottomnav .nav-link:focus,
.nav-link:hover {
    background-color: #ff658b;
    color: #fff !important;
    /* 1e1e21 */
}

.nav-tabs .nav-link.active,
.nav-tabs .nav-item.show .nav-link {
    background: linear-gradient(rgba(0, 0, 0, 0.0),
            rgba(0, 0, 0, 0.1));
    background-color: #3384db !important;
    color: #fff !important;
}

.card-header {
    background: linear-gradient(rgba(0, 0, 0, 0.0),
            rgba(0, 0, 0, 0.1));
    background-color: #3384db !important;
}

.bottombar {
    padding: 0px 0px !important;
    margin: 0px 0px !important;
    /* background-color: #f5003d !important; */
    /* background-color: #9d0027 !important; */
    font-size: 1em;
    /* font-weight: 100 !important; */
    background-color: #f13261 !important;

    /* background-color: #262626 !important; */

    box-shadow: 2px 0px 6px #000;
}

.bottombar a {
    margin: 0px 0px !important;
    padding: 5px 10px !important;
    /* color: rgba(255, 255, 255, 0.83) !important; */
    /* font-weight: 900; */

}

.corneritem a {
    width: 50px;
    background-color: #1d8cf8 !important;

    text-align: center;
    /* background-color: #0069d9 !important; */
}

.corneritem:hover a {
    background-color: #429bf3 !important;
}

.appstatus {
    background-color: #179e2d !important;
    margin: 0px 0px !important;
    padding: 5px 10px !important;
}

.appstatus-disconnected {
    color: #ffdd1d !important;
    margin: 0px 0px !important;
    padding: 5px 10px !important;
}

.navbar-nav li {
    padding: 0
}

.appname {
    font-weight: 600;
}

img .noscaleimg {
    image-rendering: optimizeSpeed;
    /* STOP SMOOTHING, GIVE ME SPEED  */
    image-rendering: -moz-crisp-edges;
    /* Firefox                        */
    image-rendering: -o-crisp-edges;
    /* Opera                          */
    image-rendering: -webkit-optimize-contrast;
    /* Chrome (and eventually Safari) */
    image-rendering: pixelated;
    /* Chrome */
    image-rendering: optimize-contrast;
    /* CSS3 Proposed                  */
    -ms-interpolation-mode: nearest-neighbor;
    /* IE8+                           */

}

img {
    -webkit-filter: drop-shadow(8px 8px 8px #000);
    filter: drop-shadow(8px 8px 8px #000);
}

.desc {

    font-size: 0.85em;
    font-weight: 300;

}

.hover:hover {
    color: white !important;
}

.page {
    display: flex;
}

.page-content {
    width: 100%;
    margin-right: 20px;
}

.help-content {
    width: 360px;

    border-left: 1px solid;
    border-color: rgb(35, 35, 36);
}

.help-content-collapsed {
    width: 30;

}

.iconlink {
    color: #dedede;
}

.outline {
    font-size: 1.3em;

}

.outline ul,
li {
    list-style-type: none;

    list-style: none;
    padding-left: 0;
}

.outline {
    margin-left: 0px;
    padding-left: 10px;
}

.outline a {
    color: #dedede;
    text-decoration: underline;
}

.outline a:hover {
    color: #fff;
}

.help-content-fullpage {
    width: 100%;
    float: left;
    display: flex;
    margin-left: 10px;

}

b {
    font-weight: 600;
}

select option {
    background-color: #000;
    border-color: #000;
}
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.1s ease;
}


.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
/* .agentfun:hover {
  -webkit-transform: scaleX(-1);
  transform: scaleX(-1);
} */
</style>

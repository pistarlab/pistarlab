<template>
<div>
    <b-button-toolbar>
        <b-button-group size="sm">
            <b-button :disabled="!launcherURI" v-b-popover.hover.top="launcherURI ? 'Start/Shutdown piSTAR Lab Services': 'Problem Loading launcher info'" :href="launcherURI" target="_blank">piSTAR Launcher</b-button>

            <b-button v-b-popover.hover.top="'Monitor Ray Tasks (link is only valid if running locally or using SSH tunneling)'" href="http://localhost:8265/" target="_blank">Ray Dashboard</b-button>
        </b-button-group>
    </b-button-toolbar>

    <hr />

    <RestartDialog></RestartDialog>
    <hr />
    <div class="mt-4"></div>
    <b-button size="sm" v-b-toggle.collapse-details variant="secondary">Configuration Details</b-button>
    <b-collapse id="collapse-details" class="mt-2">

        <div class="mt-4"></div>
        {{ message }}
        <div class="mt-4"></div>

        <hr />
        <h3>piSTAR Config</h3>
        <b-container v-if="data">
            <b-row class="pt-0" v-for="(item, name) in data.pistar_config" v-bind:key="name">
                <b-col>
                    <div>{{ name }}</div>
                </b-col>
                <b-col>

                    <div>{{ item }}</div>

                </b-col>
            </b-row>

        </b-container>

        <div class="mt-4"></div>
        <h3>available_resources</h3>
        <b-container v-if="data">
            <b-row class="pt-0" v-for="(item, name) in data.available_resources" v-bind:key="name">
                <b-col>
                    <div>{{ name }}</div>
                </b-col>
                <b-col>

                    <div>{{ item }}</div>

                </b-col>
            </b-row>
        </b-container>

        <div class="mt-4"></div>
        <h3>cluster_resources</h3>
        <b-container v-if="data">
            <b-row class="pt-0" v-for="(item, name) in data.cluster_resources" v-bind:key="name">
                <b-col>
                    <div>{{ name }}</div>
                </b-col>
                <b-col>

                    <div>{{ item }}</div>

                </b-col>
            </b-row>
        </b-container>
        <div class="mt-2"></div>
        <b-container v-if="data">
            <div class="pt-0" v-for="(item, name) in data.nodes" v-bind:key="name">
                <h6>Node: {{name}}</h6>
                <b-row>

                    <b-col>
                        <div class="pt-1 ml-4" v-for="(item, name) in item" v-bind:key="name">
                            <div>{{ name }} : {{ item }}</div>

                        </div>
                    </b-col>
                </b-row>
            </div>
        </b-container>
        <div class="mt-4"></div>
        <h3>gpu_info</h3>
        <b-container v-if="data">
            <b-row class="pt-0" v-for="(item, name) in data.gpu_info" v-bind:key="name">
                <b-col>
                    <div>{{ name }}</div>
                </b-col>
                <b-col>

                    <div>{{ item }}</div>

                </b-col>
            </b-row>
        </b-container>
        <div class="mt-2"></div>
        <h5>tensorflow_status</h5>
        <b-container v-if="data">
            <b-row class="pt-0" v-for="(item, name) in data.tensorflow_status" v-bind:key="name">
                <b-col>
                    <div>{{ name }}</div>
                </b-col>
                <b-col>

                    <div>{{ item }}</div>

                </b-col>
            </b-row>
        </b-container>
        <div class="mt-2"></div>
        <h5>torch_status</h5>
        <b-container v-if="data">
            <b-row class="pt-0" v-for="(item, name) in data.torch_status" v-bind:key="name">
                <b-col>
                    <div>{{ name }}</div>
                </b-col>
                <b-col>

                    <div>{{ item }}</div>

                </b-col>
            </b-row>
        </b-container>
    </b-collapse>
</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";
import {
    fetchSettings
} from "../app.config"
import RestartDialog from "../components/RestartDialog.vue"
export default {
    name: "Preferences",
    components: {
        // SessionList
        RestartDialog
    },
    data() {
        return {
            searchQuery: "",
            items: [],
            error: "",
            message: "",
            data: {},
            settings: {}
        };
    },
    methods: {
        adminCommand(cmd) {
            this.submitting = true;
            axios
                .get(`${this.appConfig.API_URL}/api/admin_command/` + cmd)
                .then((response) => {
                    this.message = response.data["message"];

                    console.log(response);
                    this.submitting = false;
                })
                .catch((error) => {
                    this.message = error;
                    this.submitting = false;
                });
        },

        loadAdminData() {
            axios
                .get(`${this.appConfig.API_URL}/api/admin_data/`)
                .then((response) => {
                    this.data = response.data["data"];
                })
                .catch((error) => {
                    this.message = error;
                });
        },
        loadConfig() {
            fetchSettings().then(settings => {
                this.settings = settings
            })
        },
    },
    computed: {
        launcherURI() {
            if (!this.data || !this.data.pistar_config || !this.data.pistar_config.launcher_info) {
                return null
            } else {
                return `http://${this.data.pistar_config.launcher_info.host}:${this.data.pistar_config.launcher_info.port}`
            }

        }
    },
    // Fetches posts when the component is created.
    created() {
        //

        this.loadAdminData();
        // this.loadConfig()
    },
};
</script>

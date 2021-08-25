<template>
<div>

    <div v-if="launcherURI">
        <b-button size="sm" variant="danger" :disabled="restarting" @click="restartPISTARLAB()">Restart Now</b-button>
        <span class="ml-2">NOTE: Restarting may take a while when installing new extensions.</span>
        <div class="mt-4"></div>

        <div v-if="restarting">
            <div class="mt-4"></div>
            Please wait. 
            <br />
            <b-link :href="launcherURI" target="_blank">Open Launcher Console and Log Viewer</b-link>
        </div>

        <b-alert show v-if="message" class="mt-3">
            {{message}}
            <br />
            <br />
            <b-button size="sm" v-b-toggle.collapse-details>Show Details</b-button>
            <b-collapse id="collapse-details" class="mt-2">
                <b-link :href="launcherURI" target="_blank">Open Launcher Console and Log Viewer</b-link>
                <pre>{{this.output}}</pre>
            </b-collapse>
        </b-alert>
    </div>
    <div v-else>

    </div>

</div>
</template>

<script>
import axios from "axios";

export default {
    name: "Restart",
    components: {
        // 
    },
    data() {
        return {
            message: null,
            data: {},
            restarting: false,
            output: null
        };
    },
    methods: {

        loadAdminData() {
            axios
                .get(`${this.appConfig.API_URL}/api/launcher_info`)
                .then((response) => {
                    this.data = response.data;
                })
                .catch((error) => {
                    this.message = error;
                });
        },
        restartPISTARLAB() {
            if (this.restarting || this.launcherURI == null) {
                return
            }
            this.restarting = true
            this.output = null
            this.message = null
            console.log("Restarting")
            axios
                .get(`${this.launcherURI}/api/service/restart_all`)
                .then((response) => {
                    console.log("Restart Complete")
                    this.restarting = false
                    this.message = "Restart Complete";
                    this.output = response.data

                })
                .catch((error) => {
                    console.log("Restart Failed?? Maybe")
                    this.restarting = false
                    this.message = "Restart Seems to have failed.  Please check the logs and restart manually if needed.";
                    this.output = error
                });
        }
    },
    computed: {
        launcherURI() {
            if (!this.data) {
                return null
            } else {
                return `http://${this.data.host}:${this.data.port}`
            }

        }
    },
    created() {

        this.loadAdminData();
    },
};
</script>

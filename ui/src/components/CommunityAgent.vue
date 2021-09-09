<template>
<div class="page">
    <div class="page-content">
        <h1><i class="fa fa-users"></i> Community Hub</h1>
        <h3>Agent Details</h3>

        <div v-if="item">

            <br />
            User Id: {{item.items.user_agent.user_id}}
            <br />
            Agent Name: {{item.items.user_agent.agent_name}}
            <br />
            Created: {{item.items.user_agent.created}}
            <br />
            Active snapshot Id: {{item.items.user_agent.active_snapshot_id}}

        </div>
        <div class="mt-4 h4">
            Snapshot History
        </div>
        <div class="mr-auto">
            <b-container fluid v-if="item" >
                <b-row>
                    <b-col>
                        ID
                    </b-col>
                    <b-col>
                        Agent Spec Id
                    </b-col>
                    <b-col>
                        Version
                    </b-col>
                    <b-col>
                        Downloads
                    </b-col>
                </b-row>
                <hr/>
                <div v-for="(s,i) in item.items.snapshots" v-bind:key="i">
                    <b-row>
                        <b-col>
                            {{s.snapshot_id}}
                        </b-col>
                        <b-col>
                            {{s.spec_id}}
                        </b-col>
                        <b-col>
                            {{s.version}}
                        </b-col>
                        <b-col>
                            {{s.download_counter}}
                        </b-col>
                    </b-row>
                </div>
            </b-container>

        </div>
        <div v-if="loading && !item">
            Loading...
        </div>
    </div>

</div>
</template>

<script>
import axios from "axios";

import {
    appConfig
} from "../app.config";

import {
    timedeltafordate
} from "../funcs";

export default {
    name: "AgentsOnline",
    components: {
        //
    },
    apollo: {
        //

    },
    props: {
        agentName: String,
        userId: String
    },
    data() {
        return {
            loading: true,
            item: null,
            message: null
        };
    },
    computed: {
        //

    },
    methods: {
        timedeltafordate,

        loadProfile() {
            this.loading = true
            axios
                .get(`${appConfig.API_URL}/api/online/agent_details?user_id=${this.userId}&agent_name=${this.agentName}`)
                .then((response) => {
                    this.item = response.data;
                    this.loading = false
                })
                .catch((error) => {
                    this.message = error;
                    this.loading = false
                });
        },

    },

    created() {
        this.loadProfile()

        //
    },
};
</script>

<style >
</style>

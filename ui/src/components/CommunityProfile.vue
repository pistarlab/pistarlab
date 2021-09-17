<template>
<div>
    <div v-if="data && data.user">
        <span class="h3">
            <i class="fa fa-user"></i> {{data.user.user_id}}
        </span>
        <div class="mt-5"></div>
        <h4>My Published Agents</h4>
        <div class="ml-4">
            <b-row>
                <b-col>Agent Name </b-col>
                <b-col> Last Update</b-col>
                <b-col></b-col>
            </b-row>
            <hr />
            <div v-for="(agent, i) in data.user_agents" v-bind:key="i">
                <b-row>
                    <b-col><b-link :to="`/community/agent?userId=${data.user.user_id}&agentName=${agent.agent_name}`">{{agent.agent_name}} </b-link></b-col>
                    <b-col> {{agent.updated}}</b-col>
                    <b-col></b-col>
                </b-row>
            </div>
        </div>
    </div>
    <div v-else-if="loading && !data">
        Loading...
    </div>
    <div v-else>
        <div class="mt-4" v-if="!shared.state.loggedIn">You aren't signed in.
            <b-button size="sm" title="Signed in"  v-b-modal.profile>
                Sign in/up
            </b-button>

        </div>
        <div v-else>
            Community hub is unavailable at this time.
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
    name: "CommunityProfile",
    components: {
        //
    },
    apollo: {
        //

    },
    data() {
        return {
            loading: true,
            data: null,
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
                .get(`${appConfig.API_URL}/api/online/user_details`)
                .then((response) => {
                    this.data = response.data;
                    this.message = response.data.message
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

<template>
<div>
    <div v-if="items && items.length>0">
        <b-row>
            <b-col>Agent Name </b-col>
            <b-col>User Id </b-col>
            <b-col> Last Update</b-col>
            <b-col></b-col>
        </b-row>
        <hr />
        <div v-for="(agent, i) in items" v-bind:key="i">
            <b-row>
                <b-col>
                    <b-link :to="`/community/agent?userId=${agent.user_id}&agentName=${agent.agent_name}`">{{agent.agent_name}} </b-link>
                </b-col>
                <b-col> {{agent.user_id}}</b-col>
                <b-col> {{agent.updated}}</b-col>
                <b-col></b-col>
            </b-row>
        </div>
    </div>
    <div v-else-if="loading && !items">
        Loading...
    </div>
    <div v-else>
        No Agents Found
        <br />
        {{message}}
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
    data() {
        return {
            loading: true,
            items: null,
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
                .get(`${appConfig.API_URL}/api/online/agents`)
                .then((response) => {
                    this.items = response.data.items;
                    this.loading = false
                    this.message = response.data.message
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
